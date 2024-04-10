# import streamlit as st
# import pandas as pd
# from video_analysis import compute

# def flatten_dict(d, parent_key='', sep='_'):
#     items = []
#     for k, v in d.items():
#         new_key = parent_key + sep + k if parent_key else k
#         if isinstance(v, dict):
#             items.extend(flatten_dict(v, new_key, sep=sep).items())
#         else:
#             items.append((new_key, v))
#     return dict(items)

# st.title("Video Analysis")

# uploaded_file = st.file_uploader("Upload video file", type=["mp4","mp3"])

# if uploaded_file is not None:
       
#     with st.spinner("Generating analysis..."):
#         video_info = compute(uploaded_file)
    
#     if video_info is not None:
#         st.success("Video processed successfully!")

#         flat_video_info = flatten_dict(video_info)

#         df = pd.DataFrame.from_dict(flat_video_info, orient='index', columns=['Value'])

#         st.table(df)
#     else:
#         st.error("Please upload a file.")


import streamlit as st
import json
import pandas as pd
from moviepy.editor import VideoFileClip
from video_analysis import compute
import time

def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def estimate_time(video_file):
    # Estimate time based on video file size or any other metric
    return 5  # In minutes, just a placeholder value

st.title("Video Analysis")

uploaded_file = st.file_uploader("Upload video file", type=["mp4","mp3"])

if uploaded_file is not None:
    estimated_time = estimate_time(uploaded_file)

    # Displaying estimated time
    st.write(f"Estimated time: {estimated_time} minutes")

    # Displaying a message or icon while processing is going on
    with st.spinner("Generating analysis..."):
        start_time = time.time()
        video_info = compute(uploaded_file)
        elapsed_time = time.time() - start_time

    if video_info is not None:
        st.success("Video processed successfully!")

        flat_video_info = flatten_dict(video_info)

        df = pd.DataFrame.from_dict(flat_video_info, orient='index', columns=['Value'])

        st.table(df)
    else:
        st.error("Please upload a file.")

    # Adjust the estimated time based on elapsed time if it exceeds the original estimate
    if elapsed_time < estimated_time * 60:  # Convert minutes to seconds
        remaining_time = max(estimated_time - elapsed_time / 60, 0)
        st.write(f"Estimated time: {remaining_time:.2f} minutes")
    else:
        st.write("Estimated time: Near")
