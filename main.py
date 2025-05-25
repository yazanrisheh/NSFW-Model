import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError, ImageDraw, ImageFont
from transformers import AutoModelForImageClassification, ViTImageProcessor
import torch.nn.functional as F
import plotly.express as px
import io
import pandas as pd
from datetime import datetime, timedelta
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="Falcons AI - NSFW Detection Suite", layout="wide", initial_sidebar_state="expanded")


st.markdown("""
<style>
    .stApp {
        /* Add background image or gradient here if desired */
    }
    .main-header {
        font-size: 3em;
        color: #FF4B4B; /* Falcon-like red */
        text-align: center;
        font-weight: bold;
        text-shadow: 2px 2px 4px #cccccc;
    }
    .sub-header {
        font-size: 1.8em;
        color: #333333;
        text-align: center;
        margin-bottom: 20px;
    }
    .stButton>button {
        background-color: #FF4B4B;
        color: white;
        border-radius: 20px;
        padding: 10px 20px;
        border: none;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #E04040;
    }
    .stFileUploader label {
        font-size: 1.1em !important;
    }
    .card {
        background-color: #FFFFFF;
        padding: 20px;
        border-radius: 10px;
        box_shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin_bottom: 20px;
        height: 100%; /* For consistent card height in columns */
    }
    .icon-text {
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 1.1em;
    }
    .blocked-text {
        font-weight: bold;
        color: #D32F2F; /* Red color for blocked text */
        text-align: center;
    }
    .product-card img {
        max-height: 150px;
        object-fit: cover;
        border-radius: 5px;
    }
    .timer-text {
        font-size: 1.2em;
        font-weight: bold;
        color: #FF0000; /* Bright Red for high visibility */
        text-align: center;
        padding: 15px;
        background-color: #FFEEEE; /* Light red background */
        border: 2px solid #FF4B4B;
        border-radius: 8px;
        margin-top: 10px;
        margin-bottom: 10px;
    }
    .warning-image-container {
        display: flex;
        justify-content: center;
        margin-top: 15px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)


# --- MODEL LOADING ---
@st.cache_resource
def load_model_and_processor():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "Falconsai/nsfw_image_detection"
    try:
        model = AutoModelForImageClassification.from_pretrained(model_name).to(device)
        processor = ViTImageProcessor.from_pretrained(model_name)
    except Exception as e:
        st.error(f"🚨 Error loading NSFW model/processor: {e}. Please check your internet connection and model name.")
        st.stop()
    return model, processor, device

model, processor, device = load_model_and_processor()


def run_inference(image_bytes: bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != "RGB":
            image = image.convert("RGB")
    except UnidentifiedImageError:
        return None, None, None, "UnidentifiedImageError"
    except Exception as e:
        return None, None, None, str(e)

    try:
        inputs = processor(images=image, return_tensors="pt").to(device)
    except Exception as e:
        return image, None, None, str(e)

    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    probabilities_tensor = F.softmax(logits, dim=-1)
    
    probs_list = probabilities_tensor[0].cpu().tolist()
    prob_dict = {model.config.id2label[i]: probs_list[i] for i in range(len(probs_list))}
    
    predicted_idx = logits.argmax(-1).item()
    predicted_label = model.config.id2label[predicted_idx]
    
    return image, predicted_label, prob_dict, None

def create_placeholder_image(text="Blocked", width=200, height=150, color=(200, 200, 200), text_color=(50,50,50)):
    img = Image.new('RGB', (width, height), color=color)
    d = ImageDraw.Draw(img)
    try:
        font_size = int(min(width, height) / (len(text)/2 + 4)) 
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()
    
    text_bbox = d.textbbox((0,0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    x = (width - text_width) / 2
    y = (height - text_height) / 2
    d.text((x, y), text, fill=text_color, font=font)
    return img

# --- DEMO: CONTENT FEED & FAMILY SAFETY ---
def content_feed_family_safety_demo():
    st.markdown("<div class='card'><h3>🎭 Content Feed & Family Safety</h3><p>Simulate a social feed with parental controls. Users can post content, parents can enable/disable a safety filter, and children view the filtered feed.</p></div>", unsafe_allow_html=True)

    if 'posts' not in st.session_state:
        st.session_state.posts = []
    if 'safe_mode_enabled' not in st.session_state:
        st.session_state.safe_mode_enabled = True

    tab1, tab2, tab3 = st.tabs(["👤 User: Create Post", "👨‍👩‍👧‍👦 Parent Dashboard", "🧒 Child's Feed"])

    with tab1:
        st.markdown("<h4><span class='icon-text'>✏️ Create a New Post</span></h4>", unsafe_allow_html=True)
        with st.form("new_post_form", clear_on_submit=True):
            post_caption = st.text_area("Post Caption:", placeholder="What's on your mind?", height=100)
            uploaded_image_file = st.file_uploader("Upload Image for Post:", type=["png", "jpg", "jpeg", "webp"], key="post_image_uploader")
            submit_post_button = st.form_submit_button("🚀 Create Post")

            if submit_post_button and uploaded_image_file and post_caption:
                image_bytes = uploaded_image_file.getvalue()
                processed_image, predicted_label, prob_dict, error = run_inference(image_bytes)
                
                if processed_image and predicted_label:
                    st.session_state.posts.append({
                        "caption": post_caption,
                        "image_bytes": image_bytes,
                        "image_name": uploaded_image_file.name,
                        "predicted_label": predicted_label,
                        "probabilities": prob_dict,
                        "id": len(st.session_state.posts)
                    })
                    st.success("🎉 Post created successfully!")
                else:
                    st.error(f"⚠️ Could not process image for the post. Error: {error}")
            elif submit_post_button:
                st.warning("⚠️ Please provide both a caption and an image.")
    
    with tab2:
        st.markdown("<h4><span class='icon-text'>🛡️ Parental Controls</span></h4>", unsafe_allow_html=True)
        st.session_state.safe_mode_enabled = st.toggle("Enable Safe Mode Filter", value=st.session_state.safe_mode_enabled, help="When enabled, NSFW content will be hidden in the Child's Feed.")
        st.metric("Safe Mode Status", "✅ Enabled" if st.session_state.safe_mode_enabled else "❌ Disabled")
        
        st.markdown("---")
        st.markdown("<h4><span class='icon-text'>📊 Feed Content Overview</span></h4>", unsafe_allow_html=True)
        if not st.session_state.posts:
            st.info("No posts yet. Create some posts in the 'User: Create Post' tab.")
        else:
            nsfw_count = sum(1 for post in st.session_state.posts if post['predicted_label'].lower() == 'nsfw')
            sfw_count = len(st.session_state.posts) - nsfw_count
            
            col_metrics1, col_metrics2 = st.columns(2)
            col_metrics1.metric("Total Posts", len(st.session_state.posts))
            col_metrics2.metric("Flagged as NSFW", nsfw_count)

            if nsfw_count > 0 or sfw_count > 0:
                fig = px.pie(names=['SFW', 'NSFW'], values=[sfw_count, nsfw_count], 
                             title="Content Classification Breakdown",
                             color_discrete_map={'SFW':'#63C5DA', 'NSFW':'#FF6B6B'})
                fig.update_layout(legend_title_text='Category')
                st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("📜 View All Posts (Admin View)"):
                for post in reversed(st.session_state.posts): # Newest first
                    st.markdown(f"**Post ID:** {post['id']} | **Caption:** {post['caption']}")
                    img_display = Image.open(io.BytesIO(post['image_bytes']))
                    st.image(img_display, caption=f"{post['image_name']} - Predicted: {post['predicted_label'].upper()}", width=150)
                    
                    nsfw_p = post['probabilities'].get('nsfw', 0.0) * 100
                    st.metric(label="NSFW Confidence", value=f"{nsfw_p:.2f}%", 
                                delta=f"{nsfw_p:.1f}% NSFW" if nsfw_p > 1 else None,
                                delta_color="inverse" if nsfw_p > 50 else ("off" if nsfw_p <=1 else "normal"))
                    st.markdown("---")

    with tab3:
        st.markdown("<h4><span class='icon-text'>🧸 Viewing Feed as Child</span></h4>", unsafe_allow_html=True)
        if st.session_state.safe_mode_enabled:
            st.success("🛡️ Safe Mode is ON. Potentially inappropriate content will be hidden.")
        else:
            st.warning("⚠️ Safe Mode is OFF. All content will be visible.")

        if not st.session_state.posts:
            st.info("The feed is empty. Ask a user to create some posts!")
        else:
            st.markdown(f"**Displaying {len(st.session_state.posts)} post(s):**")
            for post in reversed(st.session_state.posts):
                is_nsfw = post['predicted_label'].lower() == 'nsfw'
                
                with st.container():
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    if st.session_state.safe_mode_enabled and is_nsfw:
                        st.markdown("<p class='blocked-text'><i>Content Blocked by Parental Controls</i></p>", unsafe_allow_html=True)
                        blocked_img = create_placeholder_image("Content Blocked", width=300, height=200)
                        st.image(blocked_img, caption="Image Blocked", use_container_width=True)
                    else:
                        st.markdown(f"<blockquote>{post['caption']}</blockquote>", unsafe_allow_html=True)
                        img_display = Image.open(io.BytesIO(post['image_bytes']))
                        st.image(img_display, caption=post['image_name'], use_container_width=True)
                        if is_nsfw:
                             st.warning(f"👁️‍🗨️ This content is classified as {post['predicted_label'].upper()} ({post['probabilities'].get('nsfw',0)*100:.1f}%).")
                        else:
                             st.success(f"✅ This content is classified as {post['predicted_label'].upper()} ({post['probabilities'].get('sfw',0)*100:.1f}%).")
                    st.markdown("</div>", unsafe_allow_html=True)
                    st.markdown("---")

# --- DEMO: E-COMMERCE ---
PREMADE_PRODUCTS = [
    {"id": 1, "name": "Classic Leather Wallet", "category": "Fashion", "desc": "A timeless bifold wallet crafted from genuine leather.", "img_url": "https://picsum.photos/seed/wallet/300/200"},
    {"id": 2, "name": "Wireless Bluetooth Headphones", "category": "Electronics", "desc": "Immersive sound quality with long battery life.", "img_url": "https://picsum.photos/seed/headphones/300/200"},
    {"id": 3, "name": "Abstract Canvas Art Print", "category": "Home Goods", "desc": "Vibrant colors to brighten any room. 24x36 inches.", "img_url": "https://picsum.photos/seed/artprint/300/200"},
]

def initialize_ecommerce_session():
    if 'ecommerce_users' not in st.session_state:
        st.session_state.ecommerce_users = {
            "User1": {"flag_count": 0, "ban_until": None, "approved_products": [], "last_submission_message": None, "last_submission_message_image": None},
            "User2": {"flag_count": 0, "ban_until": None, "approved_products": [], "last_submission_message": None, "last_submission_message_image": None}
        }
    if 'ecommerce_admin_log' not in st.session_state:
        st.session_state.ecommerce_admin_log = []

def display_timer_and_message(user_name):
    user_data = st.session_state.ecommerce_users[user_name]
    ban_until_from_session = user_data.get("ban_until")

    if ban_until_from_session and datetime.now() < ban_until_from_session:
        remaining_time = ban_until_from_session - datetime.now()
        total_seconds = max(0, remaining_time.total_seconds())
        minutes = int(total_seconds // 60)
        seconds = int(total_seconds % 60)
        
        st.markdown(f"<p class='timer-text'>🚫 {user_name} is temporarily banned due to repeated NSFW uploads!<br>Try again in: {minutes:02d} min {seconds:02d} sec</p>", unsafe_allow_html=True)
        st_autorefresh(interval=1000, limit=None, key=f"timer_refresh_{user_name}")
        return True 
    
    elif ban_until_from_session and datetime.now() >= ban_until_from_session:
        st.session_state.ecommerce_users[user_name]["ban_until"] = None
        st.session_state.ecommerce_users[user_name]["flag_count"] = 0 
        st.session_state.ecommerce_users[user_name]["last_submission_message"] = {"type": "success", "text": f"✅ {user_name}, your temporary ban has ended. Please upload responsibly."}
        st.session_state.ecommerce_users[user_name]["last_submission_message_image"] = None # Clear any lingering image
        st.rerun() 
        return False 

    return False 

def handle_product_upload_form(user_name):
    st.markdown(f"<h4><span class='icon-text'>➕ List a New Product ({user_name})</span></h4>", unsafe_allow_html=True)
    
    if st.session_state.ecommerce_users[user_name]["last_submission_message"]:
        msg_type = st.session_state.ecommerce_users[user_name]["last_submission_message"]["type"]
        msg_text = st.session_state.ecommerce_users[user_name]["last_submission_message"]["text"]
        if msg_type == "success":
            st.success(msg_text)
        elif msg_type == "warning":
            st.warning(msg_text)
        elif msg_type == "error":
            st.error(msg_text)
        st.session_state.ecommerce_users[user_name]["last_submission_message"] = None

    # Display the warning image if it was set for the first strike
    if st.session_state.ecommerce_users[user_name].get("last_submission_message_image"):
        st.markdown("<div class='warning-image-container'>", unsafe_allow_html=True)
        st.image(st.session_state.ecommerce_users[user_name]["last_submission_message_image"], use_container_width=False)
        st.markdown("</div>", unsafe_allow_html=True)
        st.session_state.ecommerce_users[user_name]["last_submission_message_image"] = None

    if display_timer_and_message(user_name): 
        return 

    user_data = st.session_state.ecommerce_users[user_name]

    with st.form(f"new_product_form_{user_name}", clear_on_submit=True):
        product_name = st.text_input("Product Name", placeholder="e.g., Handcrafted Coffee Mug", key=f"pname_{user_name}")
        product_category = st.selectbox("Product Category", 
                                        ["Fashion", "Electronics", "Home Goods", "Art & Collectibles", "Handmade", "Other"], key=f"pcat_{user_name}")
        product_description = st.text_area("Product Description", placeholder="Describe your product...", height=100, key=f"pdesc_{user_name}")
        uploaded_image_file = st.file_uploader("Upload Product Image:", 
                                             type=["png", "jpg", "jpeg", "webp"],
                                             key=f"pimg_{user_name}")
        submit_button = st.form_submit_button("✅ Submit Product for Moderation")

        if submit_button:
            if not all([product_name, product_category, product_description, uploaded_image_file]):
                st.session_state.ecommerce_users[user_name]["last_submission_message"] = {"type": "warning", "text": "⚠️ Please fill in all product details and upload an image."}
                st.rerun()
                return

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            image_bytes = uploaded_image_file.getvalue()
            processed_image, predicted_label, prob_dict, error = run_inference(image_bytes)

            log_entry = {
                "User": user_name, "Product Name": product_name,
                "Image Name": uploaded_image_file.name, "Timestamp": timestamp
            }

            if processed_image and predicted_label:
                log_entry["Status"] = predicted_label.upper()
                log_entry["NSFW Score (%)"] = f"{prob_dict.get('nsfw', 0.0) * 100:.2f}"
                
                if predicted_label.lower() == "nsfw":
                    user_data["flag_count"] += 1 
                    
                    if user_data["flag_count"] == 1:
                        st.session_state.ecommerce_users[user_name]["last_submission_message"] = {"type": "warning", "text": f"🚨 PRODUCT REJECTED! Your image for '{product_name}' was flagged as NSFW ({prob_dict.get('nsfw',0)*100:.1f}%)."}
                        st.session_state.ecommerce_users[user_name]["last_submission_message_image"] = create_placeholder_image("1st Warning!", width=300, height=100, color=(255, 255, 224), text_color=(100,100,0)) 
                    elif user_data["flag_count"] >= 2:
                        user_data["ban_until"] = datetime.now() + timedelta(minutes=1) # Ban duration
                        st.session_state.ecommerce_users[user_name]["last_submission_message"] = {"type": "error", "text": f"🚩 FINAL WARNING, {user_name}! You have been flagged {user_data['flag_count']} times. You are temporarily banned."}
                        st.session_state.ecommerce_users[user_name]["last_submission_message_image"] = None 
                else: 
                    st.session_state.ecommerce_users[user_name]["last_submission_message"] = {"type": "success", "text": f"✅ Product '{product_name}' submitted and approved!"}
                    st.session_state.ecommerce_users[user_name]["last_submission_message_image"] = None 
                    user_data["approved_products"].append({
                        "name": product_name, "category": product_category, "description": product_description,
                        "image_bytes": image_bytes, "image_name": uploaded_image_file.name,
                        "sfw_score": prob_dict.get('sfw', 0.0) * 100
                    })
            else:
                st.session_state.ecommerce_users[user_name]["last_submission_message"] = {"type": "error", "text": f"🚨 Could not process the uploaded image for '{product_name}'. Error: {error}"}
                st.session_state.ecommerce_users[user_name]["last_submission_message_image"] = None
                log_entry["Status"] = "Processing Error"
                log_entry["NSFW Score (%)"] = "N/A"
            
            st.session_state.ecommerce_admin_log.append(log_entry)
            st.rerun() 

def ecommerce_demo():
    st.markdown("<div class='card'><h3>🛍️ E-commerce Product Moderation</h3><p>Users can list products. Admins monitor submissions and user behavior (e.g., temporary bans for repeated NSFW uploads).</p></div>", unsafe_allow_html=True)
    initialize_ecommerce_session()

    st.markdown("<h4><span class='icon-text'>✨ Featured Store Products </span></h4>", unsafe_allow_html=True)
    cols_featured = st.columns(len(PREMADE_PRODUCTS))
    for i, product in enumerate(PREMADE_PRODUCTS):
        with cols_featured[i]:
            st.markdown("<div class='card product-card'>", unsafe_allow_html=True)
            st.image(product["img_url"], caption=product["name"], use_container_width=True)
            st.markdown(f"**{product['name']}**")
            st.caption(f"Category: {product['category']}")
            st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("---")

    admin_tab, user1_tab, user2_tab = st.tabs(["👑 Admin Panel", "👤 User1's Storefront", "👤 User2's Storefront"])

    with admin_tab:
        st.markdown("<h4><span class='icon-text'>📊 Moderation Dashboard</span></h4>", unsafe_allow_html=True)
        
        user_stats_data = []
        for user_name_key, data_val in st.session_state.ecommerce_users.items():
            user_stats_data.append({"User": user_name_key, "Metric": "Flagged NSFW", "Count": data_val["flag_count"]})
        
        if user_stats_data:
            df_user_stats = pd.DataFrame(user_stats_data)
            if not df_user_stats.empty:
                 fig_user_stats = px.bar(df_user_stats, x="User", y="Count",
                                    title="User NSFW Flag Counts",
                                    labels={"Count": "Number of NSFW Flags"},
                                    color_discrete_sequence=["#d62728"]) 
                 st.plotly_chart(fig_user_stats, use_container_width=True)
        else:
            st.info("No user NSFW flag activity yet to visualize.")

        st.markdown("<h5><span class='icon-text'>👥 User Status</span></h5>", unsafe_allow_html=True)
        for user, data in st.session_state.ecommerce_users.items():
            is_banned = data.get("ban_until") and datetime.now() < data.get("ban_until")
            ban_status = "Banned" if is_banned else "Active"
            status_color = "red" if is_banned else "green"
            
            status_text = f"- **{user}**: Flags: {data['flag_count']}, Status: <span style='color:{status_color}; font-weight:bold;'>{ban_status}</span>"
            if is_banned:
                remaining_time = data["ban_until"] - datetime.now()
                total_seconds = max(0, remaining_time.total_seconds())
                minutes = int(total_seconds // 60)
                seconds = int(total_seconds % 60)
                status_text += f" (until {minutes:02d}m {seconds:02d}s)"
            st.markdown(status_text, unsafe_allow_html=True)


        st.markdown("<h5><span class='icon-text'>📜 Submission Log</span></h5>", unsafe_allow_html=True)
        if st.session_state.ecommerce_admin_log:
            df_log = pd.DataFrame(st.session_state.ecommerce_admin_log)
            columns_to_display = ["User", "Product Name", "Image Name", "Status", "NSFW Score (%)", "Timestamp"]
            existing_columns_to_display = [col for col in columns_to_display if col in df_log.columns]
            st.dataframe(df_log[existing_columns_to_display].sort_values(by="Timestamp", ascending=False), use_container_width=True)
        else:
            st.info("No product submission attempts yet in this session.")

    with user1_tab:
        handle_product_upload_form("User1")
        st.markdown("---")
        st.markdown("<h5><span class='icon-text'>🛍️ User1's Approved Listings</span></h5>", unsafe_allow_html=True)
        if st.session_state.ecommerce_users["User1"]["approved_products"]:
            for prod in st.session_state.ecommerce_users["User1"]["approved_products"]:
                with st.container(): 
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.image(Image.open(io.BytesIO(prod['image_bytes'])), caption=prod['image_name'], width=150)
                    st.markdown(f"**{prod['name']}** ({prod['category']}) - SFW: {prod['sfw_score']:.1f}%")
                    st.caption(prod['description'])
                    st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("User1 has not listed any SFW products yet.")


    with user2_tab:
        handle_product_upload_form("User2")
        st.markdown("---")
        st.markdown("<h5><span class='icon-text'>🛍️ User2's Approved Listings</span></h5>", unsafe_allow_html=True)
        if st.session_state.ecommerce_users["User2"]["approved_products"]:
            for prod in st.session_state.ecommerce_users["User2"]["approved_products"]:
                with st.container():
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.image(Image.open(io.BytesIO(prod['image_bytes'])), caption=prod['image_name'], width=150)
                    st.markdown(f"**{prod['name']}** ({prod['category']}) - SFW: {prod['sfw_score']:.1f}%")
                    st.caption(prod['description'])
                    st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("User2 has not listed any SFW products yet.")

# --- DEMO: BULK CLOUD STORAGE SCAN ---
def cloud_scan_demo():
    st.markdown("<div class='card'><h3>☁️ Bulk Cloud Storage Scan</h3><p>Simulate scanning a large number of images from a cloud storage service for policy violations, organization, or content auditing.</p></div>", unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader("Upload images in bulk (simulates files from cloud storage):", 
                                      type=["png", "jpg", "jpeg", "webp"], 
                                      accept_multiple_files=True,
                                      key="cloud_uploader_bulk")
    if uploaded_files:
        scan_results = []
        nsfw_count = 0
        sfw_count = 0
        error_count = 0
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        total_files = len(uploaded_files)

        for i, file in enumerate(uploaded_files):
            status_text.text(f"⚙️ Processing image {i+1}/{total_files}: {file.name}")
            image_bytes = file.getvalue()
            processed_image, predicted_label, prob_dict, error = run_inference(image_bytes)

            if processed_image and predicted_label:
                nsfw_score = prob_dict.get('nsfw', 0.0) * 100
                if predicted_label.lower() == "nsfw":
                    nsfw_count += 1
                    status_icon = "🔞"
                else:
                    sfw_count += 1
                    status_icon = "🖼️"
                scan_results.append({
                    "File Name": file.name,
                    "Prediction": f"{status_icon} {predicted_label.upper()}",
                    "NSFW Score (%)": f"{nsfw_score:.2f}",
                })
            else:
                error_count +=1
                scan_results.append({
                    "File Name": file.name, "Prediction": "🚨 Error", 
                    "NSFW Score (%)": "N/A", "Details": error
                })
            progress_bar.progress((i + 1) / total_files)
        
        status_text.success(f"✅ Processing complete for {total_files} images.")
        
        st.markdown("<h4><span class='icon-text'>📊 Scan Summary</span></h4>", unsafe_allow_html=True)
        m_cols = st.columns(4)
        m_cols[0].metric(label="Total Scanned", value=total_files)
        m_cols[1].metric(label="✅ Safe Images", value=sfw_count)
        m_cols[2].metric(label="🔞 NSFW Images", value=nsfw_count, delta=str(nsfw_count) + " flagged" if nsfw_count > 0 else None, delta_color="inverse" if nsfw_count > 0 else "normal")
        m_cols[3].metric(label="🚨 Errors", value=error_count)

        if sfw_count > 0 or nsfw_count > 0:
            fig_pie = px.pie(names=['SFW', 'NSFW'], values=[sfw_count, nsfw_count], 
                             title="Overall Content Classification",
                             color_discrete_map={"SFW": "#2ECC71", "NSFW": "#E74C3C"}) 
            st.plotly_chart(fig_pie, use_container_width=True)

        if scan_results:
            with st.expander("📜 Detailed Scan Report"):
                df_display = pd.DataFrame(scan_results)
                st.dataframe(df_display, use_container_width=True)
    else:
        st.info("📤 Upload multiple images to simulate a bulk scan of cloud storage.")


try:
    logo_image = Image.open("falcon_logo.jpeg") 
    st.sidebar.image(logo_image, width=120)
except FileNotFoundError:
    st.sidebar.warning("Logo 'falcon_logo.jpeg' not found.")
    st.sidebar.markdown("## Falcons AI", unsafe_allow_html=True) 

st.sidebar.title("✨ FalconsAI NSFW Model")
st.sidebar.markdown("---")

DEMO_OPTIONS = {
    "🎭 Content Feed & Family Safety": content_feed_family_safety_demo,
    "🛍️ E-commerce Product Moderation": ecommerce_demo,
    "☁️ Bulk Cloud Storage Scan": cloud_scan_demo,
}

selected_demo_name = st.sidebar.radio(
    "🚀 Choose a Use Case:",
    list(DEMO_OPTIONS.keys()),
    captions=[
        "Social media simulation with parental controls.",
        "User-based product moderation for online stores.",
        "Scan large image batches from cloud storage."
    ]
)

st.markdown("<h1 class='main-header'>NSFW Image Detection</h1>", unsafe_allow_html=True)
st.markdown(f"<h2 class='sub-header'>{selected_demo_name}</h2>", unsafe_allow_html=True)
st.markdown("---")

if selected_demo_name:
    demo_function = DEMO_OPTIONS[selected_demo_name]
    demo_function()

st.markdown("---")
st.markdown(
    """
    <div style='text-align:center; margin-top:40px; padding: 20px; background-color:#0E1117; border-radius:10px; color: #FAFAFA;'>
        <p style='font-size:1.1em; margin-bottom:10px;'>
            Powered by <a href='https://huggingface.co/Falconsai' target='_blank' style='color: #FF4B4B; text-decoration: none;'>Falcons.AI</a>'s 
            <a href='https://huggingface.co/Falconsai/nsfw_image_detection' target='_blank' style='color: #FF4B4B; text-decoration: none;'>NSFW Detection Model</a>
        </p>
        <p style='font-size:1em; color:#A0A0A0;'>
            🏆 Ranked #1 on Hugging Face with 100M+ Downloads 🏆
        </p>
        <p style='font-size:0.9em; color:#808080;'>
            This interactive suite demonstrates real-world applications of advanced NSFW image detection technology.
            <br>© 2024 Falcons AI. All rights reserved.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)