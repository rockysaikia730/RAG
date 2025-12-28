import streamlit as st
import uuid
import time # Import time for a quick spinner demo
import hashlib
import RAG.indexing as indexing
import RAG.model_api as model_api
import threading

import RAG.data_ingestion as data_ingest

def get_file_hash(file_object):
    file_bytes = file_object.read()
    hasher = hashlib.sha256()
    hasher.update(file_bytes)
    file_object.seek(0) 
    return hasher.hexdigest()

def update_model():
    model_name = st.session_state.selected_model
    st.session_state.model_llm = model_api.load_model(model_name=model_name)
    st.toast(f"Switched model to {model_name}", icon="🤖")

def process_uploaded_files(uploader_key, chat_to_update):
    docs = st.session_state[uploader_key]
    if not docs:
        return

    # 1. Run your logic
    existing_names_set = set(chat_to_update["doc_names"])
    new_names_set = {doc.name for doc in docs}
    newly_added_count = len(new_names_set.difference(existing_names_set))
    
    all_unique_names_set = existing_names_set.union(new_names_set)
    chat_to_update["doc_names"] = list(all_unique_names_set)

    # 2. Show a toast
    if newly_added_count > 0:
        st.toast(f"Added {newly_added_count} new file(s).", icon="✅")
    else:
        st.toast("Files are already in this chat.", icon="ℹ️")

def main():
    st.set_page_config(
        page_title="Green Horizon Genie",
        page_icon="🍃",
        layout="centered",
        initial_sidebar_state="collapsed"
    )

    # --- Inject CSS for Scrollable Chat Window ---
    st.markdown("""
        <style>
        .chat-container {
            /* Calculates height to fill viewport minus title, dividers, etc.
            You may need to adjust the 300px value for your layout.
            */
            height: calc(100vh - 800px); 
            min-height: 0px;
            overflow-y: auto; /* Enable vertical scrolling */
            display: flex;
            
            /* --- THIS IS THE CHANGE --- */
            flex-direction: column; /* Oldest at top, newest at bottom */
            /* --- END OF CHANGE --- */
            
            border: 1px solid rgba(0,0,0,0.1);
            border-radius: 8px;
            padding: 10px;
            margin-bottom: 10px;
        }

        </style>
    """, unsafe_allow_html=True)

    # --- Session State Initialization ---
    if "chats" not in st.session_state:
        st.session_state.chats = []
    if "active_chat_id" not in st.session_state:
        st.session_state.active_chat_id = None
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "qwen/qwen3-32b" # Default model
    if "model_llm" not in st.session_state:
        st.session_state.model_llm = model_api.load_model(model_name=st.session_state.selected_model )
    if "fusion_retriever" not in st.session_state:
        st.session_state.fusion_retriever = None
    if 'processing_status' not in st.session_state:
        st.session_state.processing_status = 'idle'  # idle, running, completed, error
        st.session_state.processed_documents = None
        st.session_state.processing_error = None
        st.session_state.processing_thread = None

    # --- Helper Function: Create New Chat ---
    def create_new_chat():
        chat_id = str(uuid.uuid4())
        new_chat = {
            "id": chat_id,
            "title": "New Chat",
            "doc_names": [], 
            "directory_path": None, 
            "messages": [], 
            "model": st.session_state.selected_model 
        }
        st.session_state.chats.insert(0, new_chat) 
        st.session_state.active_chat_id = chat_id

    # --- Helper Function: Get Active Chat ---
    def get_active_chat():
        if not st.session_state.active_chat_id:
            return None
        for chat in st.session_state.chats:
            if chat["id"] == st.session_state.active_chat_id:
                return chat
        return None

    # --- Sidebar: Chat Management ---
    with st.sidebar:
        st.title("Green Horizon")
        
        if st.button("New Chat ➕", key="sidebar_new_chat", use_container_width=True):
            create_new_chat()
            st.rerun() 
            
        st.header("Conversations")
        if not st.session_state.chats:
            st.caption("No chats started yet.")
        
        for chat in st.session_state.chats:
            chat_type = "primary" if chat["id"] == st.session_state.active_chat_id else "secondary"
            if st.button(chat["title"], key=chat["id"], use_container_width=True, type=chat_type):
                st.session_state.active_chat_id = chat["id"]
                st.rerun() 

    # --- Main Application ---
    st.title("🍃 Green Horizon Genie")
    st.divider()

    # --- 3. Chat Area ---
    if not st.session_state.active_chat_id and not st.session_state.chats:
        create_new_chat() 

    active_chat = get_active_chat()

    if active_chat:
        # --- Chat Interface ---
        st.header(f"{active_chat['title']}")
        st.divider()

        # --- Display existing messages (Wrapped in our CSS container) ---
        #st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        # Loop in NORMAL order to show oldest messages at the top
        for message in active_chat["messages"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                # if "source" in message:
                #     st.info(f"**Source:** {message['source']}")
        st.markdown('</div>', unsafe_allow_html=True)
        # --- End of message display ---
        
        # if active_chat["doc_names"]:
        #     doc_list = "- " + "\n- ".join(active_chat["doc_names"])
        if active_chat.get("directory_path"):
            st.markdown(f"**Active Directory:** `{active_chat['directory_path']}`")

        
        if user_question := st.chat_input("Ask your question..."):
            
            data_source_exists = active_chat["doc_names"] or active_chat.get("directory_path")
            
            if not data_source_exists:
                st.warning("Please add a data source first using the 'Add Data 📤' button! 📑")
            else:
                active_chat["messages"].append({"role": "user", "content": user_question})
                
                with st.spinner(f"Thinking with {active_chat.get('model')}... ☕"):
                    time.sleep(1) 
                    import RAG.query_eng as qe
                    import asyncio
                    response = asyncio.run(
                                        qe.rag_pipeline(user_question, st.session_state.model_llm)
                                    )
                    real_answer = response
                    #real_source = "Source from RAG pipeline."
                
                active_chat["messages"].append({
                    "role": "assistant",
                    "content": real_answer
                    #"source": real_source
                })
                
                if len(active_chat["messages"]) == 2: 
                    active_chat["title"] = user_question[:50] + "..."
                
                st.rerun()

        # --- Sticky Bottom Bar (for CONTROLS only) ---
        with st._bottom:
            col1, col2, col3 = st.columns([0.3, 0.3, 0.4])

            with col1:
                st.selectbox(
                    "Select Model (for new chats)",
                    ["qwen/qwen3-32b" , "Gemini", "llama-3.1-8b-instant"],
                    key="selected_model",
                    label_visibility="collapsed",
                    on_change=update_model 
                )
                
            with col2:
                if st.button("New Chat ➕", key="main_new_chat", use_container_width=True):
                    create_new_chat()
                    st.rerun()

            with col3:
                with st.popover("Add Data 📤", use_container_width=True):
                    st.markdown("**Add data to this chat**")
                    tab1, tab2 = st.tabs(["Connect Directory","Upload Files",])

                    with tab2:
                        docs = st.file_uploader(
                            "Upload files",
                            accept_multiple_files=True,
                            label_visibility="collapsed",
                        )
                        if st.button("Load Files",use_container_width=True):
                            if docs:  
                                with st.spinner(f"Processing Files..."):
                                    if "doc_data" not in active_chat:
                                        active_chat["doc_data"] = {}

                                    existing_hashes = set(active_chat["doc_data"].keys())
                                    new_files_to_process = []
                                    for doc in docs:
                                        # Calculate hash for each file
                                        doc_hash = get_file_hash(doc)
                                        
                                        if doc_hash not in existing_hashes:
                                            # This is a new file based on its content
                                            active_chat["doc_data"][doc_hash] = doc.name
                                            new_files_to_process.append(doc)

                                    if new_files_to_process:
                                        loaded_documents = []
                                        import tempfile
                                        import os
                                        dir_name = f"my_custom_temp_{uuid.uuid4()}"
                                        base_temp_path = tempfile.gettempdir()
                                        temp_dir = os.path.join(base_temp_path, dir_name)
                                        os.mkdir(temp_dir)

                                        file_paths = [] 
                                        
                                        for new_doc in new_files_to_process:
                                            path = os.path.join(temp_dir, new_doc.name)
                                            # Write the file's bytes to the temp path
                                            with open(path, "wb") as f:
                                                f.write(new_doc.getbuffer())
                                            file_paths.append(path)

                                        input_dir = temp_dir
                                        status = st.session_state.processing_status
                                        if status == 'idle':
                                            if os.path.isdir(input_dir):
                                                st.session_state.processing_status = 'running'
                                                st.session_state.processed_documents = None
                                                st.session_state.processing_error = None
                                                print("Starting processing thread...")
                                                thread = threading.Thread(
                                                    target=data_ingest.run_processing_in_thread,
                                                    args=(input_dir,)
                                                )
                                                st.session_state.processing_thread = thread
                                                thread.start()
                                                st.rerun()
                                            else:
                                                st.error(f"Directory not found: {input_dir}")

                                        elif status == 'running':
                                            thread = st.session_state.get('processing_thread')
                                            with st.spinner("Processing documents"):
                                                if thread:
                                                    if thread.is_alive():
                                                        time.sleep(5)
                                                    st.rerun()
                                                
                                            st.warning("Processing in progress. The UI is responsive.")
                                            

                                        elif status == 'completed':
                                            docs = st.session_state.processed_documents
                                        

                                        elif status == 'error':
                                            st.error(f"An error occurred during processing:")
                                            st.code(st.session_state.processing_error)
                                        import shutil
                                        shutil.rmtree(temp_dir)
                                                    
                                                
                                        indexing.create_or_update_retriever(loaded_documents)

                                st.toast("Files Added!")
                                active_chat["doc_names"].extend(new_files_to_process)

                                time.sleep(0.5)
                                st.rerun()
                            else:
                                st.warning("Please choose files to upload first.")
                            

                    with tab1:
                        dir_path = st.text_input(
                            "Directory Path", 
                            value=active_chat.get("directory_path", ""),
                            placeholder="e.g., /path/to/documents"
                        )
                        if st.button("Load Directory"):
                            if dir_path:
                                with st.spinner(f"Processing directory..."):
                                    pass 
                                active_chat["directory_path"] = dir_path
                                st.success(f"Connected to directory.")
                                st.rerun()
                            else:
                                st.warning("Please enter a path.")

    else:
        if st.session_state.chats:
            st.info("Select a chat from the sidebar to begin.")
        else:
            st.info("Click 'New Chat ➕' in the sidebar to get started!")

if __name__ == "__main__":
    main()