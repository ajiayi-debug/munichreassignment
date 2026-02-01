import ssl
import certifi
import os
import streamlit as st
import json  
import chromadb
from sentence_transformers import SentenceTransformer
import torch
from google import genai
from google.genai.types import HttpOptions
from dotenv import load_dotenv

ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

load_dotenv()

DB_PATH = "./chroma_db"
COLLECTION_NAME = "public_health"
GEMINI_API_KEY = os.getenv("Gemini")

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

SAMPLE_QUESTIONS = [
    "What are the main ways to fight disease germs according to the book?",
    "How does the book describe the importance of pure air and its effect on health?",
    "Based on the principles described in the book, explain why preventing germs from entering the body and maintaining a clean environment together are more effective than either measure alone in reducing disease."
]


@st.cache_resource
def load_resources():
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection(name=COLLECTION_NAME)
    embed_model = SentenceTransformer('all-mpnet-base-v2', device=DEVICE)
    return collection, embed_model


def retrieve_chunks(query: str, collection, embed_model, top_k: int = 5, ocr_source: str = "both"):
    query_embedding = embed_model.encode(query).tolist()
    
    where_filter = None
    if ocr_source == "easyocr":
        where_filter = {"source": "easyocr"}
    elif ocr_source == "qwen_vl":
        where_filter = {"source": "qwen_vl"}
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where_filter,
        include=["documents", "metadatas", "distances"]
    )
    
    chunks = []
    distances = results.get('distances', [[]])[0]
    for i in range(len(results['ids'][0])):
        distance = distances[i] if i < len(distances) else 1.0
        similarity = max(0, min(1, 1 - distance))
        
        chunks.append({
            'chunk_id': results['ids'][0][i],
            'page': results['metadatas'][0][i]['page'],
            'source': results['metadatas'][0][i].get('source', 'unknown'),
            'text': results['documents'][0][i],
            'similarity': round(similarity, 3)
        })
    
    return chunks


def construct_prompt(query: str, chunks: list) -> str:
    context = "\n\n---\n\n".join([
        f"[Page {chunk['page']}]\n{chunk['text']}" 
        for chunk in chunks
    ])
    
    prompt = f"""You are an expert assistant helping to answer questions about the book "Principles of Public Health".

Based ONLY on the following excerpts from the book, answer the question below. Your answer should be:
- Clear and concise
- Faithful to the content in the book excerpts
- Grounded in the evidence provided
- Do NOT add information not present in the excerpts

Book Excerpts:
{context}

Question: {query}

Instructions:
1. Answer the question using ONLY information from the excerpts above
2. Be specific and cite relevant details from the text
3. If the excerpts don't contain enough information to fully answer, say so
4. Keep your answer focused and well-organized
5. **UNCERTAINTY HIGHLIGHTING**: Mark any uncertain or inferred parts with [UNCERTAIN: reason]. For example: "The book suggests X [UNCERTAIN: text is partially unclear]" or "This likely means Y [UNCERTAIN: inferred from context]"
6. At the END of your answer, add a self-assessment line: "CONFIDENCE: X/10" where X is your confidence (1-10) that your answer is correct and well-supported by the excerpts.

Answer:"""
    
    return prompt


def query_gemini(prompt: str) -> str:
    client = genai.Client(
        api_key=GEMINI_API_KEY,
        http_options=HttpOptions(api_version="v1")
    )
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return response.text


def parse_confidence_and_uncertainty(answer: str):
    import re
    
    llm_confidence = None
    confidence_match = re.search(r'CONFIDENCE:\s*(\d+)/10', answer, re.IGNORECASE)
    if confidence_match:
        llm_confidence = int(confidence_match.group(1)) / 10.0
    
    uncertainties = re.findall(r'\[UNCERTAIN:\s*([^\]]+)\]', answer)
    clean_answer = re.sub(r'\n*CONFIDENCE:\s*\d+/10\s*$', '', answer, flags=re.IGNORECASE).strip()
    highlighted_answer = re.sub(
        r'\[UNCERTAIN:\s*([^\]]+)\]',
        r'⚠️ *[\1]*',
        clean_answer
    )
    
    return {
        'clean_answer': clean_answer,
        'highlighted_answer': highlighted_answer,
        'llm_confidence': llm_confidence,
        'uncertainties': uncertainties
    }


def calculate_overall_confidence(chunks: list, llm_confidence: float = None):
    if chunks:
        avg_similarity = sum(c.get('similarity', 0.5) for c in chunks) / len(chunks)
        top_similarity = chunks[0].get('similarity', 0.5) if chunks else 0.5
    else:
        avg_similarity = 0
        top_similarity = 0
    
    retrieval_score = (avg_similarity * 0.4 + top_similarity * 0.6)
    
    if llm_confidence is not None:
        overall = (retrieval_score * 0.5) + (llm_confidence * 0.5)
    else:
        overall = retrieval_score
    
    return {
        'overall': round(overall, 2),
        'retrieval': round(retrieval_score, 2),
        'llm_self_assessment': round(llm_confidence, 2) if llm_confidence else None
    }


def query_rag(query: str, collection, embed_model, top_k: int = 5, ocr_source: str = "easyocr"):
    chunks = retrieve_chunks(query, collection, embed_model, top_k, ocr_source)
    prompt = construct_prompt(query, chunks)
    raw_answer = query_gemini(prompt)
    parsed = parse_confidence_and_uncertainty(raw_answer)
    confidence = calculate_overall_confidence(chunks, parsed['llm_confidence'])
    result = {
        "question": query,
        "answer": parsed['highlighted_answer'],
        "raw_answer": parsed['clean_answer'],
        "supporting_chunks": chunks,
        "confidence": confidence,
        "uncertainties": parsed['uncertainties']
    }
    
    return result


def main():
    st.set_page_config(
        page_title="Public Health RAG Chatbot",
        page_icon="🏥",
        layout="wide"
    )
    
    st.title("🏥 Public Health RAG Chatbot")
    st.markdown("""
    Ask questions about **"Principles of Public Health"** by Thomas D. Tuttle (1910).  
    The system retrieves relevant passages from the book and generates grounded answers.
    """)
    
    with st.spinner("Loading embedding model and vector database..."):
        collection, embed_model = load_resources()
    
    with st.sidebar:
        st.header("⚙️ Settings")
        
        ocr_source = st.radio(
            "📄 OCR Source",
            options=["easyocr", "qwen_vl_paragraph", "qwen_vl_page", "compare"],
            format_func=lambda x: {
                "easyocr": "📄 EasyOCR (page)",
                "qwen_vl_paragraph": "🤖 Qwen VL (paragraph)",
                "qwen_vl_page": "📑 Qwen VL (page)",
                "compare": "⚖️ Compare All"
            }[x],
            index=0,
            help="Choose OCR source/chunking or compare all side-by-side"
        )
        
        top_k = st.slider("Number of chunks to retrieve", min_value=1, max_value=10, value=5)
        
        st.divider()
        
        st.header("📝 Sample Questions")
        st.markdown("Click a question to try it:")
        
        for i, q in enumerate(SAMPLE_QUESTIONS, 1):
            if st.button(f"Q{i}: {q[:50]}...", key=f"sample_{i}", use_container_width=True):
                st.session_state.selected_question = q
        
        st.divider()
        
        st.header("ℹ️ About")
        st.markdown("""
        This RAG system:
        - **EasyOCR** - Local OCR extraction
        - **Qwen VL** - Vision-Language model OCR
        - **ChromaDB** for vector storage
        - **all-mpnet-base-v2** for embeddings
        - **Gemini 2.5 Flash** for answer generation
        """)
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "selected_question" in st.session_state:
        query = st.session_state.selected_question
        del st.session_state.selected_question
        
        st.session_state.messages.append({"role": "user", "content": query})
        
        if ocr_source == "compare":
            with st.spinner("Querying all OCR sources..."):
                result_easyocr = query_rag(query, collection, embed_model, top_k, "easyocr")
                result_qwen_para = query_rag(query, collection, embed_model, top_k, "qwen_vl_paragraph")
                result_qwen_page = query_rag(query, collection, embed_model, top_k, "qwen_vl_page")
            
            st.session_state.messages.append({
                "role": "assistant",
                "mode": "compare",
                "easyocr": result_easyocr,
                "qwen_vl_paragraph": result_qwen_para,
                "qwen_vl_page": result_qwen_page
            })
        else:
            with st.spinner("Thinking..."):
                result = query_rag(query, collection, embed_model, top_k, ocr_source)
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": result["answer"],
                "chunks": result["supporting_chunks"],
                "confidence": result["confidence"],
                "uncertainties": result["uncertainties"]
            })
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message.get("mode") == "compare":
                st.markdown("⚖️ **Comparison Mode**: Answers from all 3 OCR sources/chunking methods")
                
                col1, col2, col3 = st.columns(3)
                
                # EasyOCR column
                with col1:
                    st.markdown("### 📄 EasyOCR (page)")
                    result = message["easyocr"]
                    st.markdown(result["answer"])
                    
                    conf = result["confidence"]
                    overall = conf.get('overall', 0)
                    color = "🟢" if overall >= 0.7 else "🟡" if overall >= 0.4 else "🔴"
                    st.markdown(f"{color} **Confidence: {overall:.0%}**")
                    
                    with st.expander("📚 Chunks"):
                        for chunk in result["supporting_chunks"]:
                            sim = chunk.get('similarity', 0)
                            st.markdown(f"**Page {chunk['page']}** | {sim:.0%}")
                            st.text(chunk["text"][:250] + "..." if len(chunk["text"]) > 250 else chunk["text"])
                            st.divider()
                
                with col2:
                    st.markdown("### 🤖 Qwen (para)")
                    result = message["qwen_vl_paragraph"]
                    st.markdown(result["answer"])
                    
                    conf = result["confidence"]
                    overall = conf.get('overall', 0)
                    color = "🟢" if overall >= 0.7 else "🟡" if overall >= 0.4 else "🔴"
                    st.markdown(f"{color} **Confidence: {overall:.0%}**")
                    
                    with st.expander("📚 Chunks"):
                        for chunk in result["supporting_chunks"]:
                            sim = chunk.get('similarity', 0)
                            st.markdown(f"**Page {chunk['page']}** | {sim:.0%}")
                            st.text(chunk["text"][:250] + "..." if len(chunk["text"]) > 250 else chunk["text"])
                            st.divider()
                
                with col3:
                    st.markdown("### 📑 Qwen (page)")
                    result = message["qwen_vl_page"]
                    st.markdown(result["answer"])
                    
                    conf = result["confidence"]
                    overall = conf.get('overall', 0)
                    color = "🟢" if overall >= 0.7 else "🟡" if overall >= 0.4 else "🔴"
                    st.markdown(f"{color} **Confidence: {overall:.0%}**")
                    
                    with st.expander("📚 Chunks"):
                        for chunk in result["supporting_chunks"]:
                            sim = chunk.get('similarity', 0)
                            st.markdown(f"**Page {chunk['page']}** | {sim:.0%}")
                            st.text(chunk["text"][:250] + "..." if len(chunk["text"]) > 250 else chunk["text"])
                            st.divider()
            else:
                st.markdown(message.get("content", ""))
                
                if message["role"] == "assistant":
                    if "confidence" in message and message["confidence"]:
                        conf = message["confidence"]
                        overall = conf.get('overall', 0)
                        
                        if overall >= 0.7:
                            color = "🟢"
                            label = "High"
                        elif overall >= 0.4:
                            color = "🟡"
                            label = "Medium"
                        else:
                            color = "🔴"
                            label = "Low"
                        
                        st.markdown(f"{color} **Confidence: {overall:.0%}** ({label})")
                        
                        with st.expander("📊 Confidence Breakdown"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Retrieval Score", f"{conf.get('retrieval', 0):.0%}")
                            with col2:
                                llm_score = conf.get('llm_self_assessment')
                                st.metric("LLM Self-Assessment", f"{llm_score:.0%}" if llm_score else "N/A")
                    
                    if "uncertainties" in message and message["uncertainties"]:
                        with st.expander(f"⚠️ Uncertainties ({len(message['uncertainties'])})"):
                            for unc in message["uncertainties"]:
                                st.warning(f"⚠️ {unc}")
                    
                    if "chunks" in message:
                        with st.expander("📚 View Supporting Chunks"):
                            for chunk in message["chunks"]:
                                source_badge = "📄" if chunk.get('source') == 'easyocr' else "🤖" if chunk.get('source') == 'qwen_vl' else "❓"
                                similarity = chunk.get('similarity', 0)
                                sim_bar = "█" * int(similarity * 10) + "░" * (10 - int(similarity * 10))
                                st.markdown(f"{source_badge} **Page {chunk['page']}** | Similarity: `{sim_bar}` {similarity:.0%}")
                                st.text(chunk["text"][:500] + "..." if len(chunk["text"]) > 500 else chunk["text"])
                                st.divider()
    
    if query := st.chat_input("Ask a question about public health principles..."):
        st.session_state.messages.append({"role": "user", "content": query})
        
        with st.chat_message("user"):
            st.markdown(query)
        
        with st.chat_message("assistant"):
            if ocr_source == "compare":
                with st.spinner("Querying all 3 OCR sources for comparison..."):
                    result_easyocr = query_rag(query, collection, embed_model, top_k, "easyocr")
                    result_qwen_para = query_rag(query, collection, embed_model, top_k, "qwen_vl_paragraph")
                    result_qwen_page = query_rag(query, collection, embed_model, top_k, "qwen_vl_page")
                
                st.markdown("⚖️ **Comparison Mode**: Answers from all 3 OCR sources/chunking methods")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("### 📄 EasyOCR (page)")
                    st.markdown(result_easyocr["answer"])
                    
                    conf = result_easyocr["confidence"]
                    overall = conf.get('overall', 0)
                    color = "🟢" if overall >= 0.7 else "🟡" if overall >= 0.4 else "🔴"
                    st.markdown(f"{color} **Confidence: {overall:.0%}**")
                    
                    if result_easyocr["uncertainties"]:
                        with st.expander(f"⚠️ Uncertainties ({len(result_easyocr['uncertainties'])})"):
                            for unc in result_easyocr["uncertainties"]:
                                st.warning(f"⚠️ {unc}")
                    
                    with st.expander("📚 Chunks"):
                        for chunk in result_easyocr["supporting_chunks"]:
                            sim = chunk.get('similarity', 0)
                            st.markdown(f"**Page {chunk['page']}** | {sim:.0%}")
                            st.text(chunk["text"][:250] + "..." if len(chunk["text"]) > 250 else chunk["text"])
                            st.divider()
                
                with col2:
                    st.markdown("### 🤖 Qwen (para)")
                    st.markdown(result_qwen_para["answer"])
                    
                    conf = result_qwen_para["confidence"]
                    overall = conf.get('overall', 0)
                    color = "🟢" if overall >= 0.7 else "🟡" if overall >= 0.4 else "🔴"
                    st.markdown(f"{color} **Confidence: {overall:.0%}**")
                    
                    if result_qwen_para["uncertainties"]:
                        with st.expander(f"⚠️ Uncertainties ({len(result_qwen_para['uncertainties'])})"):
                            for unc in result_qwen_para["uncertainties"]:
                                st.warning(f"⚠️ {unc}")
                    
                    with st.expander("📚 Chunks"):
                        for chunk in result_qwen_para["supporting_chunks"]:
                            sim = chunk.get('similarity', 0)
                            st.markdown(f"**Page {chunk['page']}** | {sim:.0%}")
                            st.text(chunk["text"][:250] + "..." if len(chunk["text"]) > 250 else chunk["text"])
                            st.divider()
                
                with col3:
                    st.markdown("### 📑 Qwen (page)")
                    st.markdown(result_qwen_page["answer"])
                    
                    conf = result_qwen_page["confidence"]
                    overall = conf.get('overall', 0)
                    color = "🟢" if overall >= 0.7 else "🟡" if overall >= 0.4 else "🔴"
                    st.markdown(f"{color} **Confidence: {overall:.0%}**")
                    
                    if result_qwen_page["uncertainties"]:
                        with st.expander(f"⚠️ Uncertainties ({len(result_qwen_page['uncertainties'])})"):
                            for unc in result_qwen_page["uncertainties"]:
                                st.warning(f"⚠️ {unc}")
                    
                    with st.expander("📚 Chunks"):
                        for chunk in result_qwen_page["supporting_chunks"]:
                            sim = chunk.get('similarity', 0)
                            st.markdown(f"**Page {chunk['page']}** | {sim:.0%}")
                            st.text(chunk["text"][:250] + "..." if len(chunk["text"]) > 250 else chunk["text"])
                            st.divider()
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "mode": "compare",
                    "easyocr": result_easyocr,
                    "qwen_vl_paragraph": result_qwen_para,
                    "qwen_vl_page": result_qwen_page
                })
            else:
                with st.spinner("Retrieving relevant passages and generating answer..."):
                    result = query_rag(query, collection, embed_model, top_k, ocr_source)
                
                st.markdown(result["answer"])
                
                conf = result["confidence"]
                overall = conf.get('overall', 0)
                
                if overall >= 0.7:
                    color = "🟢"
                    label = "High"
                elif overall >= 0.4:
                    color = "🟡"
                    label = "Medium" 
                else:
                    color = "🔴"
                    label = "Low"
                
                st.markdown(f"{color} **Confidence: {overall:.0%}** ({label})")
                
                with st.expander("📊 Confidence Breakdown"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Retrieval Score", f"{conf.get('retrieval', 0):.0%}")
                    with col2:
                        llm_score = conf.get('llm_self_assessment')
                        st.metric("LLM Self-Assessment", f"{llm_score:.0%}" if llm_score else "N/A")
                
                if result["uncertainties"]:
                    with st.expander(f"⚠️ Uncertainties ({len(result['uncertainties'])})"):
                        for unc in result["uncertainties"]:
                            st.warning(f"⚠️ {unc}")
                
                with st.expander("📚 View Supporting Chunks"):
                    for chunk in result["supporting_chunks"]:
                        source_badge = "📄" if chunk.get('source') == 'easyocr' else "🤖" if chunk.get('source') == 'qwen_vl' else "❓"
                        similarity = chunk.get('similarity', 0)
                        sim_bar = "█" * int(similarity * 10) + "░" * (10 - int(similarity * 10))
                        st.markdown(f"{source_badge} **Page {chunk['page']}** | Similarity: `{sim_bar}` {similarity:.0%}")
                        st.text(chunk["text"][:500] + "..." if len(chunk["text"]) > 500 else chunk["text"])
                        st.divider()
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "chunks": result["supporting_chunks"],
                    "confidence": result["confidence"],
                    "uncertainties": result["uncertainties"]
                })
    
    if st.session_state.messages:
        st.divider()
        col1, col2 = st.columns([3, 1])
        
        with col2:
            export_data = []
            for i in range(0, len(st.session_state.messages), 2):
                if i + 1 < len(st.session_state.messages):
                    user_msg = st.session_state.messages[i]
                    asst_msg = st.session_state.messages[i + 1]
                    
                    if asst_msg.get("mode") == "compare":
                        export_item = {
                            "question": user_msg["content"],
                            "mode": "compare",
                            "easyocr": {
                                "answer": asst_msg["easyocr"]["answer"],
                                "confidence": asst_msg["easyocr"]["confidence"],
                                "uncertainties": asst_msg["easyocr"]["uncertainties"],
                                "supporting_chunks": [
                                    {"chunk_id": c["chunk_id"], "page": c["page"], "similarity": c.get("similarity", 0)}
                                    for c in asst_msg["easyocr"]["supporting_chunks"]
                                ]
                            }
                        }
                        if "qwen_vl_paragraph" in asst_msg:
                            export_item["qwen_vl_paragraph"] = {
                                "answer": asst_msg["qwen_vl_paragraph"]["answer"],
                                "confidence": asst_msg["qwen_vl_paragraph"]["confidence"],
                                "uncertainties": asst_msg["qwen_vl_paragraph"]["uncertainties"],
                                "supporting_chunks": [
                                    {"chunk_id": c["chunk_id"], "page": c["page"], "similarity": c.get("similarity", 0)}
                                    for c in asst_msg["qwen_vl_paragraph"]["supporting_chunks"]
                                ]
                            }
                        if "qwen_vl_page" in asst_msg:
                            export_item["qwen_vl_page"] = {
                                "answer": asst_msg["qwen_vl_page"]["answer"],
                                "confidence": asst_msg["qwen_vl_page"]["confidence"],
                                "uncertainties": asst_msg["qwen_vl_page"]["uncertainties"],
                                "supporting_chunks": [
                                    {"chunk_id": c["chunk_id"], "page": c["page"], "similarity": c.get("similarity", 0)}
                                    for c in asst_msg["qwen_vl_page"]["supporting_chunks"]
                                ]
                            }
                        if "qwen_vl" in asst_msg:
                            export_item["qwen_vl"] = {
                                "answer": asst_msg["qwen_vl"]["answer"],
                                "confidence": asst_msg["qwen_vl"]["confidence"],
                                "uncertainties": asst_msg["qwen_vl"]["uncertainties"],
                                "supporting_chunks": [
                                    {"chunk_id": c["chunk_id"], "page": c["page"], "similarity": c.get("similarity", 0)}
                                    for c in asst_msg["qwen_vl"]["supporting_chunks"]
                                ]
                            }
                        export_data.append(export_item)
                    else:
                        export_data.append({
                            "question": user_msg["content"],
                            "answer": asst_msg.get("content", ""),
                            "confidence": asst_msg.get("confidence", {}),
                            "uncertainties": asst_msg.get("uncertainties", []),
                            "supporting_chunks": [
                                {"chunk_id": c["chunk_id"], "page": c["page"], "similarity": c.get("similarity", 0)}
                                for c in asst_msg.get("chunks", [])
                            ]
                        })
            
            st.download_button(
                label="📥 Export Answers (JSON)",
                data=json.dumps(export_data, indent=2, ensure_ascii=False),
                file_name="answers.json",
                mime="application/json"
            )


if __name__ == "__main__":
    main()
