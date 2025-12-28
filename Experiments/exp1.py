import streamlit as st
from docling.datamodel.base_models import InputFormat
from docling.document_converter import ImageFormatOption, PdfFormatOption
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    EasyOcrOptions
)
from llama_index.core import SimpleDirectoryReader
import os
from llama_index.readers.docling import DoclingReader
from pathlib import Path
import pikepdf
import tempfile
import zipfile
import threading
import time

# --- Your Original Helper Functions (with @st.cache_resource) ---

@st.cache_resource
def config_docling():
    """
    Configures and returns the Docling reader and exclusion patterns.
    This function is cached by Streamlit to avoid re-initializing on every script run.
    """
    st.write("Cache: Initializing Docling configuration...") # Debug line
    #----------Docling Parser--------------
    ocr_options = EasyOcrOptions()
    ocr_options.lang = ["en"]
    ocr_options.use_gpu = True  # Note: This requires a GPU-enabled environment

    ocr_pipeline_config = PdfPipelineOptions(
        do_ocr=True,
        ocr_options=ocr_options,
        verbose=True # Enabled verbose for debugging
    )

    #-----OCR for Image and PDF--------------
    docling_format_options = {
        InputFormat.IMAGE: ImageFormatOption(
            pipeline_options=ocr_pipeline_config
        ),
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=ocr_pipeline_config
        )
    }

    docling_reader = DoclingReader(
        format_options=docling_format_options
    )

    file_extractor_map = {
        ".pdf": docling_reader,
        ".docx": docling_reader,
        ".pptx": docling_reader,
        ".xlsx": docling_reader,
        ".html": docling_reader,
        ".csv": docling_reader,

        # Image formats to be OCR'd by Docling
        ".png": docling_reader,
        ".jpg": docling_reader,
        ".jpeg": docling_reader,
        ".gif": docling_reader,
    }

    exclude_extensions = ["*.zip"]
    return file_extractor_map, exclude_extensions


def enforce_mediabox_explicit(pdf_path, output_path, default_box=[0, 0, 595, 842]):
    """Ensures every page has an explicit /MediaBox defined (even if inherited)."""
    try:
        with pikepdf.open(pdf_path, allow_overwriting_input=True) as pdf:
            for page in pdf.pages:
                box = None
                try:
                    box = page.obj.get("/MediaBox")
                    if box is None:
                        box = page.page_dict.get("/MediaBox")  # fallback
                except Exception:
                    pass

                if box is None:
                    page.obj["/MediaBox"] = pikepdf.Array(default_box)
                else:
                    # Ensure box is a valid list before creating Array
                    page.obj["/MediaBox"] = pikepdf.Array(list(box))

            pdf.save(output_path)
            print(f"✅ Rewritten MediaBoxes in: {output_path}")

    except Exception as e:
        print(f"❌ Failed to enforce MediaBox on {pdf_path}: {e}")
        # If it fails, copy the original file to the output path to not block the pipeline
        if str(pdf_path) != str(output_path):
            import shutil
            shutil.copy(pdf_path, output_path)


def fix_all_pdfs(root_dir):
    """
    Iterates through all PDFs in a directory and applies enforce_mediabox_explicit.
    """
    print(f"--- Fixing PDFs in: {root_dir} ---")
    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            if file.lower().endswith(".pdf"):
                input_path = os.path.join(dirpath, file)
                # Save with the same name, pikepdf handles this
                output_path = os.path.join(dirpath, file) 
                print(f"Fixing PDF: {input_path}")
                try:
                    enforce_mediabox_explicit(input_path, output_path)
                except Exception as e:
                    print(f"❌ Failed to fix {input_path}: {e}")
    print("--- Finished fixing PDFs ---")


def process_pipeline(input_dir):
    """
    The main (long-running) processing pipeline.
    This function should NOT be called directly from the main Streamlit thread.
    """
    print(f"--- Starting pipeline for: {input_dir} ---")
    
    # --- FIX: Call the cached function ---
    # Your original code was missing the '()'
    file_extractor_map, exclude_extensions = config_docling()
    
    # --- Run PDF fix ---
    fix_all_pdfs(input_dir)

    base_input_path = Path(input_dir).resolve()
    all_documents = []

    with tempfile.TemporaryDirectory() as temp_root:
        print(f"Created temporary root: {temp_root}")
        temp_root_path = Path(temp_root).resolve()

        extracted_at_least_one_file = False

        # --- Process Zip Files ---
        for zip_path in Path(input_dir).rglob("*.zip"):
            absolute_zip_parent = zip_path.parent.resolve()
            relative_parent_dir = absolute_zip_parent.relative_to(base_input_path)
            target_extract_dir = temp_root_path / relative_parent_dir / f"_temp_extract_{zip_path.stem}"
            target_extract_dir.mkdir(parents=True, exist_ok=True)

            print(f"  Extracting: {zip_path} -> {target_extract_dir}")
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(target_extract_dir)
                extracted_at_least_one_file = True
            except Exception as e:
                print(f"❌ Failed to extract zip {zip_path}: {e}")
                st.toast(f"Failed to extract {zip_path.name}", icon="❌")
        
        if extracted_at_least_one_file:
            print("Loading documents from extracted zip contents...")
            zip_content_reader = SimpleDirectoryReader(
                input_dir=str(temp_root_path),
                file_extractor=file_extractor_map,
                recursive=True
            )
            zip_documents = zip_content_reader.load_data(show_progress=True)
            for doc in zip_documents:
                try:
                    temp_file_path = Path(doc.metadata["file_path"]).resolve()
                    relative_doc_path = temp_file_path.relative_to(temp_root_path)
                    correct_path = base_input_path / relative_doc_path
                    doc.metadata["file_path"] = str(correct_path)
                    doc.metadata["file_name"] = correct_path.name
                except Exception as e:
                    print(f"Error correcting path for doc {doc.metadata.get('file_name')}: {e}")
                
            all_documents.extend(zip_documents)

    #-----------Non-zip Documents---------------------
    print(f"Loading non-zip files from main directory: {input_dir}")
    main_reader = SimpleDirectoryReader(
        input_dir=input_dir,
        file_extractor=file_extractor_map,
        exclude=exclude_extensions,
        recursive=True
    )
    main_documents = main_reader.load_data(show_progress=True)
    all_documents.extend(main_documents)
    
    print(f"--- Pipeline complete. Loaded {len(all_documents)} documents. ---")
    return all_documents


def run_processing_in_thread(input_dir):
    try:
        # This is the long-running call
        documents = process_pipeline(input_dir)
        
        # Store results in session state
        st.session_state.processing_status = 'completed'
        st.session_state.processed_documents = documents
        st.session_state.processing_error = None

    except Exception as e:
        # Store error in session state
        st.session_state.processing_status = 'error'
        st.session_state.processing_error = str(e)
        st.session_state.processed_documents = None
        print(f"❌ THREAD ERROR: {e}")


