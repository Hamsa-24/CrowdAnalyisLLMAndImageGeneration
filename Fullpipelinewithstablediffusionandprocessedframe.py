import streamlit as st
import cv2
from ultralytics import YOLO
import pandas as pd
import cvzone
from tracker import Tracker
import numpy as np
import json
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from PIL import Image
import torch
from streamdiffusion import DiffusionPipeline

# Initialize Streamlit
st.title("Crowd Detection and Analysis")
st.write("Upload a video for crowd detection and analysis.")

# Sidebar for Image Generation prompt
st.sidebar.title("Image Generation")
prompt = st.sidebar.text_input("Enter a prompt for image generation:")
generate_button = st.sidebar.button("Generate Image")

if generate_button and prompt:
    # Load the DiffusionPipeline model
    model_id = "streamdiffusion/your-model-name"  # Update with the actual model ID
    device = "cuda"
    
    try:
        pipe = DiffusionPipeline.from_pretrained(model_id)
        pipe.to(device)

        # Generate the image
        with torch.autocast(device):
            image = pipe(prompt, guidance_scale=8.5)["sample"][0]

        # Save and display the image
        image_path = "generated_image.png"
        image.save(image_path)
        st.sidebar.image(image_path, caption="Generated Image")
    except Exception as e:
        st.sidebar.error(f"Error generating image: {e}")

# Video upload
uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])

if uploaded_video is not None:
    # Create directory if it doesn't exist
    if not os.path.exists("uploaded_video"):
        os.makedirs("uploaded_video")

    # Save the uploaded video
    video_path = os.path.join("uploaded_video", uploaded_video.name)
    with open(video_path, "wb") as f:
        f.write(uploaded_video.getbuffer())

    # Display video
    st.video(video_path)

    # Initialize the model
    model = YOLO("yolov10s.pt")

    cap = cv2.VideoCapture(video_path)
    with open("coco.txt", "r") as my_file:
        data = my_file.read()
    class_list = data.split("\n")

    tracker = Tracker()
    cy1 = 364
    offset = 10

    peoplecount = []
    count = 0

    # Initialize parameters for optical flow
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Read the first frame
    ret, old_frame = cap.read()
    old_frame = cv2.resize(old_frame, (1020, 600))
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    # Initialize the mask for drawing optical flow
    mask = np.zeros_like(old_frame)

    # Directory for processed data
    processed_data_dir = 'processed_data'
    if not os.path.exists(processed_data_dir):
        os.makedirs(processed_data_dir)

    # JSON data structure
    json_data = []

    st.write("Processing video...")

    frame_window = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (1020, 600))
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        count += 1
        if count % 3 != 0:
            old_gray = frame_gray.copy()
            continue

        results = model(frame)
        detections_data = results[0].boxes.data
        px = pd.DataFrame(detections_data).astype("float")
        detections = []
        for index, row in px.iterrows():
            x1 = int(row[0])
            y1 = int(row[1])
            x2 = int(row[2])
            y2 = int(row[3])

            d = int(row[5])
            c = class_list[d]
            if 'person' in c:
                detections.append([x1, y1, x2, y2])

        bbox_idx = tracker.update(detections)
        for bbox in bbox_idx:
            x3, y3, x4, y4, id = bbox
            cx = int((x3 + x4) / 2)
            cy = int((y3 + y4) / 2)

            if cy1 < (cy + offset) and cy1 > (cy - offset):
                cvzone.putTextRect(frame, f'{id}', (x3, y3), 1, 1)
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)
                if peoplecount.count(id) == 0:
                    peoplecount.append(id)

            # Optical flow calculation
            p0 = np.array([[cx, cy]], dtype=np.float32)
            if old_gray is not None and len(p0) > 0:
                p1, status, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
                if p1 is not None and len(p1) > 0:
                    x, y = p1[0].ravel()
                    mask = cv2.line(mask, (cx, cy), (int(x), int(y)), (0, 255, 0), 2)
                    frame = cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)

            # Log data to JSON
            json_data.append({
                "id": id,
                "timestamp": count,
                "coordinates": [cx, cy]
            })

        frame = cv2.add(frame, mask)
        cv2.line(frame, (3, 364), (1018, 364), (255, 255, 255), 1)
        people = len(peoplecount)
        cvzone.putTextRect(frame, f'People:-{people}', (50, 60), 2, 2)

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_window.image(frame_rgb)

        old_gray = frame_gray.copy()

    cap.release()
    cv2.destroyAllWindows()

    # Enhance JSON data with additional details
    enhanced_json_data = []
    for record in json_data:
        enhanced_record = {
            "event_id": str(record["id"]),
            "system_timestamp": record["timestamp"],
            "system_datetime": "2021-07-01T00:25:20Z",  # Example datetime, should be dynamically generated
            "instance_id": "camera01",
            "event_timestamp_ms": record["timestamp"] * 1000,
            "area_id": "area001",
            "area_name": "Mall",
            "location": {"latitude": 40.712776, "longitude": -74.005974},
            "temperature": "25",  # Example temperature, should be dynamically generated
            "weather": {"humidity": "60%", "wind_speed": "5 km/h", "precipitation": "0 mm"},
            "crowd_density": len(peoplecount),
            "alert_level": "low" if len(peoplecount) < 10 else "medium",
            "fire_detection": False,
            "abnormal_behavior": "none",
            "event_description": "Crowd detected in the mall.",
            "sensor_details": {"sensor_type": "video_camera", "resolution": "1080p", "frame_rate": "30 fps"},
            "image_url": f"http://example.com/images/event_{record['id']}.jpg",
            "historical_data": [
                {"timestamp": record["timestamp"] - 100, "crowd_density": len(peoplecount) - 1},
                {"timestamp": record["timestamp"] - 200, "crowd_density": len(peoplecount) - 2}
            ],
            "predicted_data": [
                {"timestamp": record["timestamp"] + 100, "predicted_crowd_density": len(peoplecount) + 1}
            ]
        }
        enhanced_json_data.append(enhanced_record)

    # Save enhanced JSON data
    enhanced_json_file_path = f"{processed_data_dir}/enhanced_data.json"
    with open(enhanced_json_file_path, 'w') as json_file:
        json.dump(enhanced_json_data, json_file, indent=4)

    st.write("Enhanced JSON data saved.")  # This should work correctly

    # RAG and LLM Integration
    def setup_llm_and_rag(json_file_path):
        # Load and parse the JSON Document
        with open(json_file_path, 'r') as file:
            data = json.load(file)

        # Convert JSON to Document Format
        documents = [Document(page_content=json.dumps(record)) for record in data]

        # Split the Document into Chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)

        # Vectorize Text Chunks and Build FAISS Index
        FAISS_PATH = 'vector_lmstudio/faiss'
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device':'cpu'})
        db = FAISS.from_documents(texts, embeddings)
        db.save_local(FAISS_PATH)

        # Setup the Local LLM
        llm = ChatOpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

        # Build a Conversational Retrieval Chain
        qa_chain = ConversationalRetrievalChain.from_llm(llm, db.as_retriever(search_kwargs={'k':2}), return_source_documents=True)

        return qa_chain

    qa_chain = setup_llm_and_rag(enhanced_json_file_path)

    st.write("You can now ask questions about the processed data.")

    # Interactive Q&A Loop
    chat_history = []

    user_query = st.text_input("Ask a question to the document:")

    if user_query:
        result = qa_chain({'question': user_query, 'chat_history': chat_history})
        st.write('Answer to your question: ' + result['answer'] + '\n')
        chat_history.append((user_query, result['answer']))
