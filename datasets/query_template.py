"""
column names:
    Rotten: rotten_tomatoes_link,review_type,review_content,top_critic,movie_info
    Amazon: asin,reviewText,verified,overall,summary,Format,description
    Beer: beer/name,beer/beerId,beer/brewerId,beer/ABV,beer/style,review/appearance,review/aroma,review/palate,review/taste,review/overall,review/time,review/profileName,review/text
    PDMX: path,metadata,mxl,pdf,mid,version,isuserpro,isuserpublisher,isuserstaff,haspaywall,israted,isofficial,isoriginal,isdraft,hascustomaudio,hascustomvideo,ncomments,nfavorites,nviews,nratings,rating,license,licenseurl,licenseconflict,genres,groups,tags,songname,title,subtitle,artistname,composername,publisher,complexity,ntracks,tracks,songlength,songlengthseconds,songlengthbars,songlengthbeats,nnotes,notesperbar,nannotations,hasannotations,nlyrics,haslyrics,ntokens,pitchclassentropy,scaleconsistency,grooveconsistency,bestpath,isbestpath,bestarrangement,isbestarrangement,bestuniquearrangement,isbestuniquearrangement,subsetall,subsetrated,subsetdeduplicated,subsetrateddeduplicated,subsetnolicenseconflict,subsetallvalid, comment, description
"""

template_rotton = {
    "binary": "Decide whether this movie is a good choice for the whole family to watch together based on the movie information: [{movie_info}] and the user review: [{review_content}]. Output only 'Yes' or 'No' and nothing else.",
    "classify": "Based on the movie's information: [{review_content}], categorize it into one of the following categories: 'Action', 'Thriller', 'Western', 'Horror', 'Comedy', 'Drama', 'Science Fiction', 'Romance', 'Crime', 'Animation', 'Documentary', 'Fantasy', 'Musical', 'Adventure', 'Historical', 'Children', 'War', 'Music', 'Cyberpunk'. Output only the category and nothing else.",
    "regression": "Output only a number and nothing else for the following task: rate a satisfaction score between 0 (bad) and 5 (good) based on review: [{review_content}] and movie info: [{movie_info}].",
    "summarize": "Summarize the user's review: [{review_content}] on the movie: [{movie_info}].",
    "open": "Recommend some movies for the user based on the movie information: [{movie_info}] and the user's review: [{review_content}].",
}

template_amazon = {
    "binary": "Analyze whether this product would be suitable for kids based on the description: [{description}] and user review: [{reviewText}], output only 'Yes' or 'No' and nothing else.",
    "classify": "Analyze the sentiment of user's review: [{reviewText}]. Output only 'Negative', 'Positive', or 'Neutral' and nothing else.",
    "regression": "Predict the overall rating of the music product from 0 (bad) to 5 (good) based on the reviewText: [{reviewText}]. Output nothing else but only the number.",
    "summarize": "Summarize the user's review: [{reviewText}] on the product: [{description}].",
    "open": "Tell me the most likely target audience for the music product given its Format: [{Format}] and the user comment: [{reviewText}]."
}

template_beer = {
    "binary": "Based on the following descriptions: Beer's name [{beer/name}], Beer's style [{beer/style}], Beer's ABV [{beer/ABV}], does this beer have European origin? Output only 'Yes' or 'No' and nothing else.",
    "classify": "Analyze the sentiments of user's review: [{review/text}], output only 'Negative', 'Positive', or 'Neutral'.",
    "regression": "Predict user's overall rating of the beer from 0 (bad) to 20 (good) based on the reviewText: [{review/text}]. Output nothing else but only the number.",
    "summarize": "Summarize the user's review on the beer and make sure that the summarization is helpful to explain the users' rating. User's review is: [{review/text}]. The beer's information is: Beer's name [{beer/name}], Beer's style [{beer/style}], Beer's ABV [{beer/ABV}]. The user's rating on the beer is: [{review/overall}].",
    "open": "From the brewer's view, what is the most prominent problem of the beer based on the user's review and aspect-specific ratings? User's review: [{review/text}], appearence rating: {review/appearance}, aroma rating: {review/aroma}, palate rating: {review/palate}, taste rating: {review/taste}."
}

template_pdmx = {
    "binary": "Given the description [{description}] of the music [{songname}], does this song information references a specific individual? Answer only 'YES' or 'NO', nothing else.",
    "classify": "Analyze the sentiment of user's comment: [{comment}], output only 'Negative', 'Positive', or 'Neutral' and nothing else.",
    "regression": "Predict the rating stars of the music from 0 (bad) to 5 (good) based on the user's comment [{comment}] and music's genres [{genres}], description [{description}]. Output nothing else but only the number.",
    "summarize": "Summarize the user's comment: [{comment}] on the music: [{description}].",
    "open": "Tell me the most likely target audience for the music given its title [{songname}], artist name [{artistname}], composer name [{composername}], publisher [{publisher}], genres [{genres}], description [{description}], and the user comment: [{comment}]."
}

template_output_len = {
    "binary": 5,
    "classify": 10,
    "regression":  5,
    "summarize": 50,
    "open": 100
}

template = {'amazon': template_amazon, 'rotten': template_rotton, 'beer': template_beer, 'pdmx':template_pdmx}
