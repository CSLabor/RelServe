"""Benchmark online relational LLM serving.
"""
import io
import os
import sys
import copy
import time
import json
import random
import asyncio
import argparse
import warnings
from dataclasses import dataclass, asdict
from typing import Any, AsyncGenerator, Collection, Dict, List, Optional, Tuple

import numpy as np
from utils.backend_request_func import (ASYNC_REQUEST_FUNCS, RequestFuncInput,
                                  RequestFuncOutput)
from tqdm.asyncio import tqdm
from transformers import PreTrainedTokenizerBase

try:
    from vllm.transformers_utils.tokenizer import get_tokenizer
except ImportError:
    from backend_request_func import get_tokenizer

try:
    from vllm.utils import FlexibleArgumentParser
except ImportError:
    from argparse import ArgumentParser as FlexibleArgumentParser

from load_queries import relQuery_load

MILLISECONDS_TO_SECONDS_CONVERSION = 1000


@dataclass
class BenchmarkMetrics:
    completed: int
    total_input: int
    total_output: int
    request_throughput: float
    output_throughput: float
    total_token_throughput: float
    mean_ttft_ms: float
    median_ttft_ms: float
    std_ttft_ms: float
    percentiles_ttft_ms: List[Tuple[float, float]]
    mean_tpot_ms: float
    median_tpot_ms: float
    std_tpot_ms: float
    percentiles_tpot_ms: List[Tuple[float, float]]
    mean_itl_ms: float
    median_itl_ms: float
    std_itl_ms: float
    percentiles_itl_ms: List[Tuple[float, float]]
    # E2EL stands for end-to-end latency per request.
    # It is the time taken on the client side from sending
    # a request to receiving a complete response.
    mean_e2el_ms: float
    median_e2el_ms: float
    std_e2el_ms: float
    percentiles_e2el_ms: List[Tuple[float, float]]

def convert_relQuery_to_request(relQuery_list, tokenizer, max_model_len):
    """
    each request: (prompt_text, input_len, output_len, priority, rel_id)

    * the priority is used by vllm + (static) priority scheduling
    * the rel_id is used by our method, for dynamic priority scheduling

    initial priority = total #input&output tokens of the belonging relQuery
    """
    if "opt" in tokenizer.name_or_path:
        tokenizer.chat_template = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ 'Question:\n' + message['content'] + '\n\n' }}{% elif message['role'] == 'system' %}\n{{ 'System:\n' + message['content'] + '\n\n' }}{% elif message['role'] == 'assistant' %}{{ 'Answer:\n'  + message['content'] + '\n\n' }}{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ 'Answer:\n' }}{% endif %}{% endfor %}"
    requests = []
    for idx, relQuery in enumerate(relQuery_list):
        sum_prompt_tokens = 0
        sum_decode_tokens = 0
        accum_relQuery_tokens = 0
        cur_requests = []
        for request in relQuery:
            message = [
                    {'role': 'system', 'content': request["system_content"]},
                    {'role': 'user', 'content': request["user_content"]}
            ]
            prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
            prompt_token_ids = tokenizer.apply_chat_template(message, tokenize=True, add_generation_prompt=True)
            #print(prompt_token_ids)
            #sys.exit()
            input_len = len(prompt_token_ids)
            output_len = request["output_len"]
            if input_len + output_len > max_model_len - 10:
                # for too long requests, we ignore them
                continue
            cur_requests.append([prompt, input_len, output_len, None, idx])
            accum_relQuery_tokens += input_len + output_len
            sum_prompt_tokens += input_len
            sum_decode_tokens += output_len
        for req in cur_requests:
            req[-2] = accum_relQuery_tokens
            requests.append(tuple(req))
        if len(cur_requests) == 0:
            # extreme case, all requests in a relQuery have too long length
            continue
        print(f"{idx:4d}-th relQuery: {len(cur_requests):4d} requests, avg {accum_relQuery_tokens//len(cur_requests):4d} tokens, prio {accum_relQuery_tokens:6d}, avg input len {sum_prompt_tokens//len(cur_requests):4d}, avg output len {sum_decode_tokens//len(cur_requests):4d}")

    return requests


def sample_dummy_requests(count, tokenizer, input_len=300, output_len=10):
    requests = []
    for i in range(count):
        fake_inp = [i+3600] * (input_len-1)
        fake_prompt = tokenizer.decode(fake_inp)
        requests.append((fake_prompt, input_len, output_len, 0, 0))
    return requests

async def get_relational_request(
    input_requests: List[Tuple[str, int, int]],
    request_rate: float,
    burstiness: float = 1.0,
    zero_priority: bool = False,
) -> AsyncGenerator[Tuple[str, int, int], None]:
    """
    Asynchronously generates requests at a specified rate 
    with OPTIONAL burstiness.
    
    Args:
        input_requests: 
            A list of input requests, each represented as a tuple.
        request_rate: 
            The rate at which requests are generated (requests/s).
        burstiness (optional): 
            The burstiness factor of the request generation. 
            Only takes effect when request_rate is not inf.
            Default value is 1, which follows a Poisson process.
            Otherwise, the request intervals follow a gamma distribution.
            A lower burstiness value (0 < burstiness < 1) results 
            in more bursty requests, while a higher burstiness value 
            (burstiness > 1) results in a more uniform arrival of requests.
    """
    # Calculate scale parameter theta to maintain the desired request_rate.
    assert burstiness > 0, (
        f"A positive burstiness factor is expected, but given {burstiness}.")
    theta = 1.0 / (request_rate * burstiness)

    prev_rel_id = input_requests[0][-1]
    cnt = 0
    for request in input_requests:
        if request_rate == float("inf"):
            yield request
            continue

        if request[-1] != prev_rel_id:
            # a new relational query, need to sleep
            cnt += 1
            interval = np.random.gamma(shape=burstiness, scale=theta)
            print(f"\nsleep {interval} seconds before sending {cnt}-th relational query")
            await asyncio.sleep(interval)
            prev_rel_id = request[-1]

        if zero_priority:
            request = copy.deepcopy(request)
            request = list(request)
            request[-2] = 0
            request = tuple(request)
        yield request

def my_calc_metric(inputs, outputs):
    relQuery_latency_dict = dict()
    for req_id, o in enumerate(outputs):
        if not o.success:
            orig_input = inputs[req_id]
            print(f"{req_id}-th request failed (rel_id: {orig_input[-1]})")
            #print(f"{orig_input}")
            print(o.error)
            continue
            #raise ValueError
        _, _, _, _, rel_id = inputs[req_id]
        if rel_id not in relQuery_latency_dict:
            relQuery_latency_dict[rel_id] = []
        relQuery_latency_dict[rel_id].append(o.latency)

    sorted_rel_ids = sorted(relQuery_latency_dict.keys())
    query_latency_list = []
    for rel_id in sorted_rel_ids:
        cur_latency = max(relQuery_latency_dict[rel_id])
        query_latency_list.append(cur_latency)
        print(f"{rel_id:4d}-th relQuery latency: {cur_latency:.3f} ({len(relQuery_latency_dict[rel_id])} successful requests)")

    print(f"%%%% MEAN Latency={np.mean(query_latency_list):.2f} for {len(sorted_rel_ids)} relQueries %%%%")
    # TODO: save per request lantency for each relational query

def save_to_json(inputs: List[RequestFuncInput], outputs: List[RequestFuncOutput], req_trace_file_name: str):
    # Convert inputs and outputs to dictionaries
    inputs_dict = [asdict(input_item) for input_item in inputs]
    outputs_dict = [asdict(output_item) for output_item in outputs]

    # Pack inputs and outputs into one dictionary
    #data = {
    #    "inputs": inputs_dict,
    #    "outputs": outputs_dict
    #}
    data = [dict() for _ in range(len(inputs_dict))]
    for a,b in zip(data, inputs_dict):
        a.update(b)
    for a,b in zip(data, outputs_dict):
        a.update(b)

    # Convert the dictionary to a JSON string
    json_data = json.dumps(data, indent=4)

    # Write the JSON string to a file
    with open(req_trace_file_name, 'w') as json_file:
        json_file.write(json_data)


def calculate_metrics(
    input_requests: List[Tuple[str, int, int]],
    outputs: List[RequestFuncOutput],
    dur_s: float,
    tokenizer: PreTrainedTokenizerBase,
    selected_percentile_metrics: List[str],
    selected_percentiles: List[float],
) -> Tuple[BenchmarkMetrics, List[int]]:
    actual_output_lens: List[int] = []
    total_input = 0
    completed = 0
    itls: List[float] = []
    tpots: List[float] = []
    all_tpots: List[float] = []
    ttfts: List[float] = []
    e2els: List[float] = []
    for i in range(len(outputs)):
        if outputs[i].success:
            # We use the tokenizer to count the number of output tokens for all
            # serving backends instead of looking at len(outputs[i].itl) since
            # multiple output tokens may be bundled together
            # Note : this may inflate the output token count slightly
            output_len = len(
                tokenizer(outputs[i].generated_text,
                          add_special_tokens=False).input_ids)
            actual_output_lens.append(output_len)
            total_input += input_requests[i][1]
            tpot = 0
            if output_len > 1:
                tpot = (outputs[i].latency - outputs[i].ttft) / (output_len -
                                                                 1)
                tpots.append(tpot)
            # Note: if output_len <= 1, we regard tpot as 0 for goodput
            all_tpots.append(tpot)
            itls += outputs[i].itl
            ttfts.append(outputs[i].ttft)
            e2els.append(outputs[i].latency)
            completed += 1
        else:
            actual_output_lens.append(0)

    if completed == 0:
        warnings.warn(
            "All requests failed. This is likely due to a misconfiguration "
            "on the benchmark arguments.",
            stacklevel=2)
    metrics = BenchmarkMetrics(
        completed=completed,
        total_input=total_input,
        total_output=sum(actual_output_lens),
        request_throughput=completed / dur_s,
        output_throughput=sum(actual_output_lens) / dur_s,
        total_token_throughput=(total_input + sum(actual_output_lens)) / dur_s,
        mean_ttft_ms=np.mean(ttfts or 0) *
        1000,  # ttfts is empty if streaming is not supported by backend
        std_ttft_ms=np.std(ttfts or 0) * 1000,
        median_ttft_ms=np.median(ttfts or 0) * 1000,
        percentiles_ttft_ms=[(p, np.percentile(ttfts or 0, p) * 1000)
                             for p in selected_percentiles],
        mean_tpot_ms=np.mean(tpots or 0) * 1000,
        std_tpot_ms=np.std(tpots or 0) * 1000,
        median_tpot_ms=np.median(tpots or 0) * 1000,
        percentiles_tpot_ms=[(p, np.percentile(tpots or 0, p) * 1000)
                             for p in selected_percentiles],
        mean_itl_ms=np.mean(itls or 0) * 1000,
        std_itl_ms=np.std(itls or 0) * 1000,
        median_itl_ms=np.median(itls or 0) * 1000,
        percentiles_itl_ms=[(p, np.percentile(itls or 0, p) * 1000)
                            for p in selected_percentiles],
        mean_e2el_ms=np.mean(e2els or 0) * 1000,
        std_e2el_ms=np.std(e2els or 0) * 1000,
        median_e2el_ms=np.median(e2els or 0) * 1000,
        percentiles_e2el_ms=[(p, np.percentile(e2els or 0, p) * 1000)
                             for p in selected_percentiles],
    )

    return metrics, actual_output_lens



async def benchmark(
    backend: str,
    api_url: str,
    base_url: str,
    model_id: str,
    tokenizer: PreTrainedTokenizerBase,
    input_requests: List[Tuple[str, int, int, int]],
    logprobs: Optional[int],
    best_of: int,
    request_rate: float,
    burstiness: float,
    disable_tqdm: bool,
    profile: bool,
    selected_percentile_metrics: List[str],
    selected_percentiles: List[str],
    ignore_eos: bool,
    zero_priority: bool,
    req_trace_file_name: str
):
    if backend in ASYNC_REQUEST_FUNCS:
        request_func = ASYNC_REQUEST_FUNCS[backend]
    else:
        raise ValueError(f"Unknown backend: {backend}")

    print("Starting initial 10 prompt test run...")
    tos = []
    for i in range(10):
        test_prompt, test_prompt_len, test_output_len, test_prio, _ = input_requests[-i]
        if zero_priority:
            test_prio = 0
        test_input = RequestFuncInput(
            model=model_id,
            prompt=test_prompt,
            api_url=api_url,
            prompt_len=test_prompt_len,
            output_len=1,
            logprobs=logprobs,
            extra_body={'priority':test_prio, 'rel_id':123456789},
            best_of=best_of,
            ignore_eos=ignore_eos
        )
        test_output = await request_func(request_func_input=test_input)
        tos.append(test_output)
    if not all(test_output.success for test_output in tos):
        raise ValueError(
            "Initial test run failed - Please make sure benchmark arguments "
            f"are correctly specified. Error: {test_output.error for test_output in tos}")
    else:
        print("Initial test run completed. Starting main benchmark run...")
    await asyncio.sleep(10)

    if burstiness == 1.0:
        distribution = "Poisson process"
    else:
        distribution = "Gamma distribution"

    print(f"Traffic request rate: {request_rate}")
    print(f"Burstiness factor: {burstiness} ({distribution})")

    pbar = None if disable_tqdm else tqdm(total=len(input_requests))

    semaphore = None

    async def limited_request_func(request_func_input, pbar):
        if semaphore is None:
            return await request_func(request_func_input=request_func_input,
                                      pbar=pbar)
        async with semaphore:
            return await request_func(request_func_input=request_func_input,
                                      pbar=pbar)

    benchmark_start_time = time.perf_counter()
    tasks: List[asyncio.Task] = []
    record_request_inputs = []
    async for request in get_relational_request(input_requests, request_rate, burstiness, zero_priority):
        prompt, prompt_len, output_len, prio, rel_id = request
        request_func_input = RequestFuncInput(model=model_id,
                                              prompt=prompt,
                                              api_url=api_url,
                                              prompt_len=prompt_len,
                                              output_len=output_len,
                                              logprobs=logprobs,
                                              extra_body={'priority':prio, 'rel_id':rel_id},
                                              best_of=best_of,
                                              ignore_eos=ignore_eos)
        record_request_inputs.append(request_func_input)
        tasks.append(
            asyncio.create_task(
                limited_request_func(request_func_input=request_func_input,
                                     pbar=pbar)))
    outputs: List[RequestFuncOutput] = await asyncio.gather(*tasks)

    if pbar is not None:
        pbar.close()

    benchmark_duration = time.perf_counter() - benchmark_start_time

    my_calc_metric(input_requests, outputs)
    save_to_json(record_request_inputs, outputs, req_trace_file_name)

    metrics, actual_output_lens = calculate_metrics(
        input_requests=input_requests,
        outputs=outputs,
        dur_s=benchmark_duration,
        tokenizer=tokenizer,
        selected_percentile_metrics=selected_percentile_metrics,
        selected_percentiles=selected_percentiles,
    )

    print("{s:{c}^{n}}".format(s=' Serving Benchmark Result ', n=50, c='='))
    print("{:<40} {:<10}".format("Successful requests:", metrics.completed))
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):",
                                    benchmark_duration))
    print("{:<40} {:<10}".format("Total input tokens:", metrics.total_input))
    print("{:<40} {:<10}".format("Total generated tokens:",
                                 metrics.total_output))
    print("{:<40} {:<10.2f}".format("Request throughput (req/s):",
                                    metrics.request_throughput))
    print("{:<40} {:<10.2f}".format("Output token throughput (tok/s):",
                                    metrics.output_throughput))
    print("{:<40} {:<10.2f}".format("Total Token throughput (tok/s):",
                                    metrics.total_token_throughput))

    result = {
        "duration": benchmark_duration,
        "completed": metrics.completed,
        "total_input_tokens": metrics.total_input,
        "total_output_tokens": metrics.total_output,
        "request_throughput": metrics.request_throughput,
        "output_throughput": metrics.output_throughput,
        "total_token_throughput": metrics.total_token_throughput,
        "input_lens": [output.prompt_len for output in outputs],
        "output_lens": actual_output_lens,
        "ttfts": [output.ttft for output in outputs],
        "itls": [output.itl for output in outputs],
        "generated_texts": [output.generated_text for output in outputs],
        "errors": [output.error for output in outputs],
    }

    def process_one_metric(
        # E.g., "ttft"
        metric_attribute_name: str,
        # E.g., "TTFT"
        metric_name: str,
        # E.g., "Time to First Token"
        metric_header: str,
    ):
        # This function prints and adds statistics of the specified
        # metric.
        if metric_attribute_name not in selected_percentile_metrics:
            return
        print("{s:{c}^{n}}".format(s=metric_header, n=50, c='-'))
        print("{:<40} {:<10.2f}".format(
            f"Mean {metric_name} (ms):",
            getattr(metrics, f"mean_{metric_attribute_name}_ms")))
        print("{:<40} {:<10.2f}".format(
            f"Median {metric_name} (ms):",
            getattr(metrics, f"median_{metric_attribute_name}_ms")))
        result[f"mean_{metric_attribute_name}_ms"] = getattr(
            metrics, f"mean_{metric_attribute_name}_ms")
        result[f"median_{metric_attribute_name}_ms"] = getattr(
            metrics, f"median_{metric_attribute_name}_ms")
        result[f"std_{metric_attribute_name}_ms"] = getattr(
            metrics, f"std_{metric_attribute_name}_ms")
        for p, value in getattr(metrics,
                                f"percentiles_{metric_attribute_name}_ms"):
            p_word = str(int(p)) if int(p) == p else str(p)
            print("{:<40} {:<10.2f}".format(f"P{p_word} {metric_name} (ms):",
                                            value))
            result[f"p{p_word}_{metric_attribute_name}_ms"] = value

    process_one_metric("ttft", "TTFT", "Time to First Token")
    process_one_metric("tpot", "TPOT",
                       "Time per Output Token (excl. 1st token)")
    process_one_metric("itl", "ITL", "Inter-token Latency")
    process_one_metric("e2el", "E2EL", "End-to-end Latency")

    print("=" * 50)

    return result



def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    req_trace_file_name = f"req_trace/{args.model.split('/')[1]}_"
    req_trace_file_name += f"{args.dataset_name}_{args.request_rate}_{args.num_rel}_{args.min_num_req}_{args.max_num_req}"
    req_trace_file_name += f"_{args.dyn}"
    print("req_trace_file_name: ", req_trace_file_name)


    backend = args.backend
    model_id = args.model
    tokenizer_id = args.tokenizer if args.tokenizer is not None else args.model

    if args.base_url is not None:
        api_url = f"{args.base_url}{args.endpoint}"
        base_url = f"{args.base_url}"
    else:
        api_url = f"http://{args.host}:{args.port}{args.endpoint}"
        base_url = f"http://{args.host}:{args.port}"

    tokenizer = get_tokenizer(tokenizer_id,
                              trust_remote_code=args.trust_remote_code)

    if args.dataset_name == "random":
        raise ValueError
    else:
        raw_relational_queries = relQuery_load(args.dataset_name,
                                              args.dataset_path,
                                              args.num_rel,
                                              args.min_num_req,
                                              args.max_num_req)
        max_model_len = 2048 if "opt" in model_id else 8192
        input_requests = convert_relQuery_to_request(raw_relational_queries, tokenizer, max_model_len)


    benchmark_result = asyncio.run(
        benchmark(
            backend=backend,
            api_url=api_url,
            base_url=base_url,
            model_id=model_id,
            tokenizer=tokenizer,
            input_requests=input_requests,
            logprobs=args.logprobs,
            best_of=args.best_of,
            request_rate=args.request_rate,
            burstiness=args.burstiness,
            disable_tqdm=args.disable_tqdm,
            profile=args.profile,
            selected_percentile_metrics=args.percentile_metrics.split(","),
            selected_percentiles=[
                float(p) for p in args.metric_percentiles.split(",")
            ],
            ignore_eos=args.ignore_eos,
            zero_priority=args.zerop,
            req_trace_file_name=req_trace_file_name,
        ))

if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Benchmark the online relational queries serving.")
    parser.add_argument(
        "--backend",
        type=str,
        default="vllm",
        choices=list(ASYNC_REQUEST_FUNCS.keys()),
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Server or API base url if not using http host and port.",
    )
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--endpoint",
        type=str,
        default="/v1/completions",
        help="API endpoint.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="all",
        choices=["all", "amazon", "rotten", "beer", "pdmx"],
    )
    parser.add_argument("--dataset-path",
                        type=str,
                        default="./datasets"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name of the model.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        help="Name or path of the tokenizer, if not using the default tokenizer.",
    )
    parser.add_argument(
        "--best-of",
        type=int,
        default=1,
        help="Generates `best_of` sequences per prompt and "
        "returns the best one.",
    )
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument(
        "--logprobs",
        type=int,
        default=None,
        help=("Number of logprobs-per-token to compute & return as part of "
              "the request. If unspecified, then either (1) if beam search "
              "is disabled, no logprobs are computed & a single dummy "
              "logprob is returned for each token; or (2) if beam search "
              "is enabled 1 logprob per token is computed"),
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=1/10,
        help="Number of relational queries per second. If this is inf, "
        "then all the requests are sent at time 0. "
        "Otherwise, we use Poisson process or gamma distribution "
        "to synthesize the request arrival times.",
    )
    parser.add_argument(
        "--burstiness",
        type=float,
        default=1.0,
        help="Burstiness factor of the request generation. "
        "Only take effect when request_rate is not inf. "
        "Default value is 1, which follows Poisson process. "
        "Otherwise, the request intervals follow a gamma distribution. "
        "A lower burstiness value (0 < burstiness < 1) results in more "
        "bursty requests. A higher burstiness value (burstiness > 1) "
        "results in a more uniform arrival of requests.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code from huggingface",
    )
    parser.add_argument(
        "--disable-tqdm",
        action="store_true",
        help="Specify to disable tqdm progress bar.",
    )
    parser.add_argument(
        "--save-result",
        action="store_true",
        help="Specify to save benchmark results to a json file",
    )
    parser.add_argument(
        "--metadata",
        metavar="KEY=VALUE",
        nargs="*",
        help="Key-value pairs (e.g, --metadata version=0.3.3 tp=1) "
        "for metadata of this run to be saved in the result JSON file "
        "for record keeping purposes.",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default=None,
        help="Specify directory to save benchmark json results."
        "If not specified, results are saved in the current directory.",
    )
    parser.add_argument(
        "--result-filename",
        type=str,
        default=None,
        help="Specify the filename to save benchmark json results."
        "If not specified, results will be saved in "
        "{backend}-{args.request_rate}qps-{base_model_id}-{current_dt}.json"
        " format.",
    )
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help="Set ignore_eos flag when sending the benchmark request."
        "Warning: ignore_eos is not supported in deepspeed_mii and tgi.")
    parser.add_argument(
        "--percentile-metrics",
        type=str,
        default="ttft,tpot,itl",
        help="Comma-seperated list of selected metrics to report percentils. "
        "This argument specifies the metrics to report percentiles. "
        "Allowed metric names are \"ttft\", \"tpot\", \"itl\", \"e2el\". "
        "Default value is \"ttft,tpot,itl\".")
    parser.add_argument(
        "--metric-percentiles",
        type=str,
        default="99",
        help="comma-seperated list of percentiles for selected metrics. "
        "to report 25-th, 50-th, and 75-th percentiles, use \"25,50,75\". "
        "default value is \"99\". "
        "use \"--percentile-metrics\" to select metrics.",
    )
    parser.add_argument(
        "--profile",
        action="store_true"
    )

    # group for dataset specific arguments
    random_group = parser.add_argument_group("random dataset options")
    random_group.add_argument(
        "--random-input-len",
        type=int,
        default=1024,
        help=
        "Number of input tokens per request, used only for random sampling.",
    )
    random_group.add_argument(
        "--random-output-len",
        type=int,
        default=10,
        help=
        "Number of output tokens per request, used only for random sampling.",
    )
    rel_group = parser.add_argument_group("relational LLM query dataset options")
    rel_group.add_argument(
        "--num-rel",
        type=int,
        default=20,
        help=
        "Number of relational queries",
    )
    rel_group.add_argument(
        "--min-num-req",
        type=int,
        default=10,
        help=
        "Minimal number of requests per relQuery",
    )
    rel_group.add_argument(
        "--max-num-req",
        type=int,
        default=500,
        help=
        "Maximal number of requests per relQuery",
    )

    # arguments for experiments control
    parser.add_argument(
        "--zerop",
        action="store_true",
        help="change priority of all requests to zero before sending"
    )

    # arguments to store intermediate results
    parser.add_argument(
        "--dyn",
        type=str,
        required=True,
        help="backend dynamic or fixed priority.",
    )

    args = parser.parse_args()
    main(args)
