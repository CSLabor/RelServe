import random
import numpy as np
import pandas as pd

def relQuery_load(ds_name, root, query_num, min_req_num, max_req_num, shuffle=True, inner_sort=True):
    if ds_name == "all":
        candidates_ds = ['amazon', 'rotten', 'beer', 'pdmx']
    else:
        candidates_ds = [ds_name]
    dataset_list = np.random.choice(candidates_ds, (query_num,))
    nrows_list = np.random.randint(min_req_num, max_req_num+1, (query_num,))
    from datasets.query_template import template
    all_types = list(template["amazon"].keys())
    types_list = np.random.choice(all_types, (query_num,))
    return _load_rq(dataset_list, nrows_list, types_list, root, shuffle, inner_sort)

def _load_rq(dataset_list, nrows_list, types_list, root="./datasets", shuffle=True, inner_sort=True):
    system_content = ("You are a helpful assistant that answers users' questions as simple as possible.\n"
            "You need to pay attention to and strictly follow users' instructions on the output format.\n"
            "You should maintain a consistent criterion when judging or predicting the results.")
    from datasets.query_template import template, template_output_len
    amazon_df = pd.read_csv(f"{root}/Amazon_Product_Reviews.csv")
    rotten_df = pd.read_csv(f"{root}/Rotten_Tomatoes_Movie_Reviews.csv")
    beer_df = pd.read_csv(f"{root}/RateBeer.csv")
    pdmx_df = pd.read_csv(f"{root}/PDMX.csv")

    result = []
    for dataset_name, num_rows, query_type in zip(dataset_list, nrows_list, types_list):
        assert dataset_name in ['amazon', 'rotten', 'beer', 'pdmx']
        raw_df = eval(f"{dataset_name}_df")
        #assert num_rows <= raw_df.shape[0]
        num_rows = min(num_rows, len(raw_df))
        cur_df = raw_df.sample(num_rows, random_state=0)
        query_template = template[dataset_name][query_type]
        user_contents = cur_df.apply(lambda row: query_template.format(**row.to_dict()), axis=1).to_list()
        cur_query = []
        for u in user_contents:
            output_len = template_output_len[query_type]
            cur_query.append({'system_content': system_content, 'user_content': u, 'output_len': output_len})
        if inner_sort:
            cur_query = sorted(cur_query, key=lambda x: x['user_content'])
        result.append(cur_query)

    if shuffle:
        random.shuffle(result)

    return result



