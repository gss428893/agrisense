# ============================================================
# SIH26131 - AgriSense (Exact Pesticide Name + Dosage, 7 Languages, Offline)
# ============================================================

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

st.set_page_config(page_title="AgriSense", page_icon="🌾", layout="centered")

# ---------- Language Switch ----------
LANGUAGES = ["हिंदी (Hindi)", "मराठी (Marathi)", "ਪੰਜਾਬੀ (Punjabi)", "తెలుగు (Telugu)", "தமிழ் (Tamil)", "বাংলা (Bengali)", "English"]
LANG_CODES = ["hi", "mr", "pa", "te", "ta", "bn", "en"]

if "lang" not in st.session_state:
    st.session_state.lang = "hi"

col1, col2 = st.columns([2, 1])
with col2:
    choice = st.selectbox("Language / भाषा", LANGUAGES, index=LANG_CODES.index(st.session_state.lang))
    st.session_state.lang = LANG_CODES[LANGUAGES.index(choice)]

LANG = st.session_state.lang

# ---------- UI Text ----------
TEXT = {
    "title": {"hi": "🌾 एग्रीसेंस", "mr": "🌾 अ‍ॅग्रीसेन्स", "pa": "🌾 ਐਗਰੀਸੈਂਸ", "te": "🌾 అగ్రిసెన్స్", "ta": "🌾 அக்ரிசென்ஸ்", "bn": "🌾 এগ্রিসেন্স", "en": "🌾 AgriSense"},
    "subtitle": {
        "hi": "AI आधारित फसल रोग व दवा सुझाव प्रणाली", "mr": "AI आधारित पीक रोग व औषध सूचना प्रणाली",
        "pa": "AI ਅਧਾਰਿਤ ਫਸਲ ਰੋਗ ਤੇ ਦਵਾਈ ਸੁਝਾਅ ਸਿਸਟਮ", "te": "AI ఆధారిత పంట వ్యాధి & మందు సూచన వ్యవస్థ",
        "ta": "AI அடிப்படையிலான பயிர் நோய் & மருந்து பரிந்துரை அமைப்பு", "bn": "AI ভিত্তিক ফসলের রোগ ও ওষুধ পরামর্শ ব্যবস্থা",
        "en": "AI-Powered Crop Disease & Pesticide Advisor"
    },
    "tagline": {
        "hi": "SIH26131 | पूरी तरह ऑफलाइन काम करता है", "mr": "SIH26131 | पूर्णपणे ऑफलाइन कार्य करते",
        "pa": "SIH26131 | ਪੂਰੀ ਤਰ੍ਹਾਂ ਆਫਲਾਈਨ ਕੰਮ ਕਰਦਾ ਹੈ", "te": "SIH26131 | పూర్తిగా ఆఫ్‌లైన్‌లో పనిచేస్తుంది",
        "ta": "SIH26131 | முழுமையாக ஆஃப்லைனில் செயல்படுகிறது", "bn": "SIH26131 | সম্পূর্ণ অফলাইনে কাজ করে",
        "en": "SIH26131 | Works Fully Offline"
    },
    "upload_label": {
        "hi": "पत्ती की फोटो अपलोड करें (केवल पत्ती, फल/सब्जी नहीं)", "mr": "पानाचा फोटो अपलोड करा (फक्त पान, फळ/भाजी नाही)",
        "pa": "ਪੱਤੇ ਦੀ ਫੋਟੋ ਅੱਪਲੋਡ ਕਰੋ (ਸਿਰਫ਼ ਪੱਤਾ, ਫਲ/ਸਬਜ਼ੀ ਨਹੀਂ)", "te": "ఆకు ఫోటోను అప్‌లోడ్ చేయండి (ఆకు మాత్రమే)",
        "ta": "இலை புகைப்படத்தை பதிவேற்றவும் (இலை மட்டும்)", "bn": "পাতার ছবি আপলোড করুন (শুধু পাতা)",
        "en": "Upload a leaf photo (leaf only, not fruit/tuber)"
    },
    "camera_label": {
        "hi": "या सीधे कैमरे से फोटो लें", "mr": "किंवा थेट कॅमेऱ्याने फोटो घ्या",
        "pa": "ਜਾਂ ਸਿੱਧਾ ਕੈਮਰੇ ਤੋਂ ਫੋਟੋ ਲਓ", "te": "లేదా నేరుగా కెమెరాతో ఫోటో తీయండి",
        "ta": "அல்லது நேரடியாக கேமராவில் புகைப்படம் எடுக்கவும்", "bn": "অথবা সরাসরি ক্যামেরা থেকে ছবি তুলুন",
        "en": "Or take a photo directly"
    },
    "uploaded_caption": {"hi": "अपलोड की गई फोटो", "mr": "अपलोड केलेला फोटो", "pa": "ਅੱਪਲੋਡ ਕੀਤੀ ਫੋਟੋ", "te": "అప్‌లోడ్ చేసిన ఫోటో", "ta": "பதிவேற்றப்பட்ட புகைப்படம்", "bn": "আপলোড করা ছবি", "en": "Uploaded Image"},
    "analyzing": {
        "hi": "AI फसल की जांच कर रहा है...", "mr": "AI पिकाची तपासणी करत आहे...",
        "pa": "AI ਫਸਲ ਦੀ ਜਾਂਚ ਕਰ ਰਿਹਾ ਹੈ...", "te": "AI పంటను పరిశీలిస్తోంది...",
        "ta": "AI பயிரை பரிசோதிக்கிறது...", "bn": "AI ফসল পরীক্ষা করছে...",
        "en": "AI is analyzing the crop..."
    },
    "result_heading": {"hi": "🔍 परिणाम", "mr": "🔍 निकाल", "pa": "🔍 ਨਤੀਜਾ", "te": "🔍 ఫలితం", "ta": "🔍 முடிவு", "bn": "🔍 ফলাফল", "en": "🔍 Result"},
    "confidence_label": {"hi": "विश्वसनीयता", "mr": "विश्वासार्हता", "pa": "ਭਰੋਸੇਯੋਗਤਾ", "te": "విశ్వసనీయత", "ta": "நம்பகத்தன்மை", "bn": "নির্ভরযোগ্যতা", "en": "Confidence"},
    "severity_label": {"hi": "गंभीरता", "mr": "तीव्रता", "pa": "ਗੰਭੀਰਤਾ", "te": "తీవ్రత", "ta": "தீவிரம்", "bn": "তীব্রতা", "en": "Severity"},
    "pesticide_label": {"hi": "🧪 दवा का नाम", "mr": "🧪 औषधाचे नाव", "pa": "🧪 ਦਵਾਈ ਦਾ ਨਾਮ", "te": "🧪 మందు పేరు", "ta": "🧪 மருந்தின் பெயர்", "bn": "🧪 ওষুধের নাম", "en": "🧪 Pesticide Name"},
    "dosage_label": {"hi": "📏 कितनी मात्रा में डालें", "mr": "📏 किती प्रमाणात टाकावे", "pa": "📏 ਕਿੰਨੀ ਮਾਤਰਾ ਵਿੱਚ ਪਾਓ", "te": "📏 ఎంత మోతాదు వేయాలి", "ta": "📏 எவ்வளவு அளவு போடவேண்டும்", "bn": "📏 কতটা পরিমাণে দিন", "en": "📏 Dosage"},
    "frequency_label": {"hi": "🔁 कितनी बार छिड़कें", "mr": "🔁 किती वेळा फवारणी करा", "pa": "🔁 ਕਿੰਨੀ ਵਾਰ ਛਿੜਕੋ", "te": "🔁 ఎన్నిసార్లు పిచికారీ చేయాలి", "ta": "🔁 எத்தனை முறை தெளிக்க வேண்டும்", "bn": "🔁 কতবার স্প্রে করবেন", "en": "🔁 Spray Frequency"},
    "verified_badge": {"hi": "✅ ICAR/विश्वविद्यालय अनुशंसित", "mr": "✅ ICAR/विद्यापीठ शिफारसीत", "pa": "✅ ICAR/ਯੂਨੀਵਰਸਿਟੀ ਸਿਫ਼ਾਰਸ਼ੀ", "te": "✅ ICAR/విశ్వవిద్యాలయ సిఫార్సు", "ta": "✅ ICAR/பல்கலைக்கழக பரிந்துரை", "bn": "✅ ICAR/বিশ্ববিদ্যালয় প্রস্তাবিত", "en": "✅ ICAR/University Verified"},
    "estimate_badge": {"hi": "⚠️ अनुमानित मात्रा - विशेषज्ञ से पुष्टि करें", "mr": "⚠️ अंदाजे प्रमाण - तज्ञांकडून खात्री करा", "pa": "⚠️ ਅਨੁਮਾਨਿਤ ਮਾਤਰਾ - ਮਾਹਿਰ ਤੋਂ ਪੁਸ਼ਟੀ ਕਰੋ", "te": "⚠️ అంచనా మోతాదు - నిపుణుడితో నిర్ధారించుకోండి", "ta": "⚠️ மதிப்பிடப்பட்ட அளவு - நிபுணரிடம் உறுதிப்படுத்தவும்", "bn": "⚠️ আনুমানিক পরিমাণ - বিশেষজ্ঞের সাথে নিশ্চিত করুন", "en": "⚠️ Estimated dosage - please confirm with local expert"},
    "no_cure_badge": {"hi": "❌ कोई सीधी दवा नहीं - पौधा हटाना जरूरी", "mr": "❌ थेट औषध नाही - रोपटे काढणे आवश्यक", "pa": "❌ ਕੋਈ ਸਿੱਧੀ ਦਵਾਈ ਨਹੀਂ - ਪੌਧਾ ਹਟਾਉਣਾ ਜ਼ਰੂਰੀ", "te": "❌ ప్రత్యక్ష మందు లేదు - మొక్కను తొలగించాలి", "ta": "❌ நேரடி மருந்து இல்லை - செடியை அகற்ற வேண்டும்", "bn": "❌ সরাসরি ওষুধ নেই - গাছ সরানো প্রয়োজন", "en": "❌ No direct cure - plant removal needed"},
    "low_confidence_warning": {
        "hi": "⚠️ फोटो साफ नहीं है या पत्ती स्पष्ट नहीं दिख रही। कृपया नज़दीक से, अच्छी रोशनी में दोबारा फोटो लें।",
        "mr": "⚠️ फोटो स्पष्ट नाही किंवा पान नीट दिसत नाही. कृपया जवळून, चांगल्या प्रकाशात पुन्हा फोटो घ्या.",
        "pa": "⚠️ ਫੋਟੋ ਸਾਫ਼ ਨਹੀਂ ਹੈ। ਕਿਰਪਾ ਕਰਕੇ ਨੇੜਿਓਂ, ਚੰਗੀ ਰੋਸ਼ਨੀ ਵਿੱਚ ਦੁਬਾਰਾ ਫੋਟੋ ਲਓ।",
        "te": "⚠️ ఫోటో స్పష్టంగా లేదు. దయచేసి దగ్గరగా, మంచి వెలుతురులో మళ్ళీ ఫోటో తీయండి.",
        "ta": "⚠️ புகைப்படம் தெளிவாக இல்லை. தயவுசெய்து அருகில், நல்ல வெளிச்சத்தில் மீண்டும் எடுக்கவும்.",
        "bn": "⚠️ ছবিটি স্পষ্ট নয়। অনুগ্রহ করে কাছ থেকে, ভালো আলোতে আবার ছবি তুলুন।",
        "en": "⚠️ Photo is unclear or the leaf isn't clearly visible. Please retake the photo closer, in good light."
    },
    "waiting_msg": {
        "hi": "👆 ऊपर फोटो अपलोड करें या कैमरे से लें", "mr": "👆 वर फोटो अपलोड करा किंवा कॅमेऱ्याने घ्या",
        "pa": "👆 ਉੱਪਰ ਫੋਟੋ ਅੱਪਲੋਡ ਕਰੋ ਜਾਂ ਕੈਮਰੇ ਤੋਂ ਲਓ", "te": "👆 పైన ఫోటో అప్‌లోడ్ చేయండి",
        "ta": "👆 மேலே புகைப்படத்தை பதிவேற்றவும்", "bn": "👆 উপরে ছবি আপলোড করুন",
        "en": "👆 Upload a photo above or use the camera"
    },
    "disclaimer": {
        "hi": "अस्वीकरण: यह SIH26131 के लिए बनाया गया प्रोटोटाइप है। दवा का प्रयोग करने से पहले नज़दीकी कृषि विशेषज्ञ या कृषि विज्ञान केंद्र (KVK) से मात्रा की पुष्टि अवश्य करें। लेबल निर्देश हमेशा पढ़ें।",
        "mr": "अस्वीकरण: हे SIH26131 साठी तयार केलेले प्रोटोटाइप आहे. औषध वापरण्यापूर्वी जवळच्या कृषी तज्ञ किंवा KVK कडून प्रमाणाची खात्री करा.",
        "pa": "ਬੇਦਾਅਵਾ: ਇਹ SIH26131 ਲਈ ਬਣਾਇਆ ਪ੍ਰੋਟੋਟਾਈਪ ਹੈ। ਦਵਾਈ ਵਰਤਣ ਤੋਂ ਪਹਿਲਾਂ ਮਾਤਰਾ ਦੀ ਪੁਸ਼ਟੀ ਨੇੜਲੇ KVK ਤੋਂ ਕਰੋ।",
        "te": "నిరాకరణ: ఇది SIH26131 కోసం రూపొందించిన ప్రోటోటైప్. మందు వాడే ముందు మోతాదును సమీప KVK వద్ద నిర్ధారించుకోండి.",
        "ta": "மறுப்பு: இது SIH26131க்காக உருவாக்கப்பட்ட முன்மாதிரி. மருந்தைப் பயன்படுத்தும் முன் அளவை உள்ளூர் KVK-இல் உறுதிப்படுத்தவும்.",
        "bn": "দাবিত্যাগ: এটি SIH26131-এর জন্য তৈরি একটি প্রোটোটাইপ। ওষুধ ব্যবহারের আগে পরিমাণ নিকটস্থ KVK-তে নিশ্চিত করুন।",
        "en": "Disclaimer: This is a prototype built for SIH26131. Always confirm exact dosage with a local agriculture expert or Krishi Vigyan Kendra (KVK) before applying. Always read the product label."
    },
    "sidebar_title": {"hi": "एग्रीसेंस के बारे में", "mr": "अ‍ॅग्रीसेन्स बद्दल", "pa": "ਐਗਰੀਸੈਂਸ ਬਾਰੇ", "te": "అగ్రిసెన్స్ గురించి", "ta": "அக்ரிசென்ஸ் பற்றி", "bn": "এগ্রিসেন্স সম্পর্কে", "en": "About AgriSense"},
}

SEVERITY_TEXT = {
    "High": {"hi": "उच्च", "mr": "जास्त", "pa": "ਉੱਚ", "te": "అధిక", "ta": "அதிகம்", "bn": "উচ্চ", "en": "High"},
    "Medium": {"hi": "मध्यम", "mr": "मध्यम", "pa": "ਦਰਮਿਆਨਾ", "te": "మధ్యస్థం", "ta": "நடுத்தரம்", "bn": "মাঝারি", "en": "Medium"},
    "None": {"hi": "कोई नहीं", "mr": "काहीही नाही", "pa": "ਕੋਈ ਨਹੀਂ", "te": "ఏమీ లేదు", "ta": "எதுவும் இல்லை", "bn": "কোনটি নয়", "en": "None"},
}

# ============================================================
# DISEASE DATABASE - Pesticide Name + Exact Dosage + Frequency
#
# "verified": True  -> ICAR / university / multi-source confirmed dosage
# "verified": False -> commonly-used estimate; app shows a warning to confirm locally
# "verified": None  -> no chemical cure exists; shows removal/control advice instead
# ============================================================

DISEASE_DB = {
    "Apple___Apple_scab": {"crop": "Apple", "disease_hi": "सेब - स्कैब रोग", "disease_en": "Apple Scab", "severity": "Medium",
        "pesticide": "Mancozeb 75% WP", "dosage_hi": "2-2.5 ग्राम प्रति लीटर पानी", "dosage_en": "2-2.5 g per litre of water",
        "freq_hi": "कली फूटने से हर 10-15 दिन में", "freq_en": "Every 10-15 days from bud break", "verified": True},
    "Apple___Black_rot": {"crop": "Apple", "disease_hi": "सेब - ब्लैक रॉट", "disease_en": "Black Rot", "severity": "High",
        "pesticide": "Captan 50% WP", "dosage_hi": "2 ग्राम प्रति लीटर पानी", "dosage_en": "2 g per litre of water",
        "freq_hi": "हर 10-14 दिन में", "freq_en": "Every 10-14 days", "verified": False},
    "Apple___Cedar_apple_rust": {"crop": "Apple", "disease_hi": "सेब - सीडर रस्ट", "disease_en": "Cedar Apple Rust", "severity": "Medium",
        "pesticide": "Mancozeb 75% WP", "dosage_hi": "2-2.5 ग्राम प्रति लीटर पानी", "dosage_en": "2-2.5 g per litre of water",
        "freq_hi": "वसंत में हर 10-14 दिन में", "freq_en": "Every 10-14 days in early spring", "verified": False},
    "Apple___healthy": {"crop": "Apple", "disease_hi": "सेब - स्वस्थ", "disease_en": "Healthy", "severity": "None",
        "pesticide": None, "dosage_hi": "आवश्यकता नहीं", "dosage_en": "Not needed", "freq_hi": "-", "freq_en": "-", "verified": None},
    "Blueberry___healthy": {"crop": "Blueberry", "disease_hi": "ब्लूबेरी - स्वस्थ", "disease_en": "Healthy", "severity": "None",
        "pesticide": None, "dosage_hi": "आवश्यकता नहीं", "dosage_en": "Not needed", "freq_hi": "-", "freq_en": "-", "verified": None},
    "Cherry_(including_sour)___Powdery_mildew": {"crop": "Cherry", "disease_hi": "चेरी - पाउडरी मिल्ड्यू", "disease_en": "Powdery Mildew", "severity": "Medium",
        "pesticide": "Sulfur 80% WP (Wettable Sulfur)", "dosage_hi": "2-3 ग्राम प्रति लीटर पानी", "dosage_en": "2-3 g per litre of water",
        "freq_hi": "हर 10-12 दिन में", "freq_en": "Every 10-12 days", "verified": False},
    "Cherry_(including_sour)___healthy": {"crop": "Cherry", "disease_hi": "चेरी - स्वस्थ", "disease_en": "Healthy", "severity": "None",
        "pesticide": None, "dosage_hi": "आवश्यकता नहीं", "dosage_en": "Not needed", "freq_hi": "-", "freq_en": "-", "verified": None},
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {"crop": "Corn/Maize", "disease_hi": "मक्का - ग्रे लीफ स्पॉट", "disease_en": "Gray Leaf Spot", "severity": "Medium",
        "pesticide": "Mancozeb 75% WP", "dosage_hi": "2.5-3 ग्राम प्रति लीटर पानी", "dosage_en": "2.5-3 g per litre of water",
        "freq_hi": "पहला लक्षण दिखते ही, हर 10 दिन में", "freq_en": "At first symptom, every 10 days", "verified": False},
    "Corn_(maize)___Common_rust_": {"crop": "Corn/Maize", "disease_hi": "मक्का - रतुआ रोग", "disease_en": "Common Rust", "severity": "Medium",
        "pesticide": "Mancozeb 75% WP (Dithane M-45)", "dosage_hi": "2.5-3 ग्राम प्रति लीटर पानी", "dosage_en": "2.5-3 g per litre of water",
        "freq_hi": "फफोले दिखते ही, जरूरत पर 15 दिन बाद दोबारा", "freq_en": "At first pustules; repeat after 15 days if severe", "verified": True},
    "Corn_(maize)___Northern_Leaf_Blight": {"crop": "Corn/Maize", "disease_hi": "मक्का - झुलसा रोग", "disease_en": "Northern Leaf Blight", "severity": "High",
        "pesticide": "Mancozeb 75% WP", "dosage_hi": "2.5-3 ग्राम प्रति लीटर पानी", "dosage_en": "2.5-3 g per litre of water",
        "freq_hi": "बुवाई के 20 दिन बाद, हर 10 दिन में", "freq_en": "20 days after sowing, every 10 days", "verified": False},
    "Corn_(maize)___healthy": {"crop": "Corn/Maize", "disease_hi": "मक्का - स्वस्थ", "disease_en": "Healthy", "severity": "None",
        "pesticide": None, "dosage_hi": "आवश्यकता नहीं", "dosage_en": "Not needed", "freq_hi": "-", "freq_en": "-", "verified": None},
    "Grape___Black_rot": {"crop": "Grape", "disease_hi": "अंगूर - ब्लैक रॉट", "disease_en": "Black Rot", "severity": "High",
        "pesticide": "Mancozeb 75% WP", "dosage_hi": "2-2.5 ग्राम प्रति लीटर पानी", "dosage_en": "2-2.5 g per litre of water",
        "freq_hi": "कली फूटने से पकने तक हर 10-14 दिन में", "freq_en": "Every 10-14 days from bud break to veraison", "verified": False},
    "Grape___Esca_(Black_Measles)": {"crop": "Grape", "disease_hi": "अंगूर - एस्का रोग", "disease_en": "Esca (Black Measles)", "severity": "High",
        "pesticide": None, "dosage_hi": "कोई सीधी दवा नहीं", "dosage_en": "No direct chemical cure",
        "freq_hi": "रोगग्रस्त शाखाएं काटकर नष्ट करें", "freq_en": "Prune and destroy infected wood; consult ICAR-NRC Grapes Pune", "verified": None},
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {"crop": "Grape", "disease_hi": "अंगूर - पत्ती झुलसा", "disease_en": "Leaf Blight", "severity": "Medium",
        "pesticide": "Copper Oxychloride 50% WP", "dosage_hi": "2.5-3 ग्राम प्रति लीटर पानी", "dosage_en": "2.5-3 g per litre of water",
        "freq_hi": "हर 10-12 दिन में", "freq_en": "Every 10-12 days", "verified": True},
    "Grape___healthy": {"crop": "Grape", "disease_hi": "अंगूर - स्वस्थ", "disease_en": "Healthy", "severity": "None",
        "pesticide": None, "dosage_hi": "आवश्यकता नहीं", "dosage_en": "Not needed", "freq_hi": "-", "freq_en": "-", "verified": None},
    "Orange___Haunglongbing_(Citrus_greening)": {"crop": "Orange/Citrus", "disease_hi": "संतरा - सिट्रस ग्रीनिंग", "disease_en": "Citrus Greening (HLB)", "severity": "High",
        "pesticide": None, "dosage_hi": "कोई इलाज नहीं", "dosage_en": "No cure available",
        "freq_hi": "पेड़ तुरंत हटाएं, वाहक कीट (साइलिड) नियंत्रित करें", "freq_en": "Remove tree immediately; control psyllid vector; contact Citrus Research Station", "verified": None},
    "Peach___Bacterial_spot": {"crop": "Peach", "disease_hi": "आड़ू - बैक्टीरियल स्पॉट", "disease_en": "Bacterial Spot", "severity": "Medium",
        "pesticide": "Copper Oxychloride 50% WP", "dosage_hi": "2.5-3 ग्राम प्रति लीटर पानी", "dosage_en": "2.5-3 g per litre of water",
        "freq_hi": "सुप्त मौसम में, हर 10-14 दिन में", "freq_en": "During dormant season, every 10-14 days", "verified": False},
    "Peach___healthy": {"crop": "Peach", "disease_hi": "आड़ू - स्वस्थ", "disease_en": "Healthy", "severity": "None",
        "pesticide": None, "dosage_hi": "आवश्यकता नहीं", "dosage_en": "Not needed", "freq_hi": "-", "freq_en": "-", "verified": None},
    "Pepper,_bell___Bacterial_spot": {"crop": "Bell Pepper", "disease_hi": "शिमला मिर्च - बैक्टीरियल स्पॉट", "disease_en": "Bacterial Spot", "severity": "Medium",
        "pesticide": "Copper Oxychloride 50% WP", "dosage_hi": "2.5-3 ग्राम प्रति लीटर पानी", "dosage_en": "2.5-3 g per litre of water",
        "freq_hi": "हर 7-10 दिन में", "freq_en": "Every 7-10 days", "verified": True},
    "Pepper,_bell___healthy": {"crop": "Bell Pepper", "disease_hi": "शिमला मिर्च - स्वस्थ", "disease_en": "Healthy", "severity": "None",
        "pesticide": None, "dosage_hi": "आवश्यकता नहीं", "dosage_en": "Not needed", "freq_hi": "-", "freq_en": "-", "verified": None},
    "Potato___Early_blight": {"crop": "Potato", "disease_hi": "आलू - अगेती झुलसा रोग", "disease_en": "Early Blight", "severity": "Medium",
        "pesticide": "Mancozeb 75% WP", "dosage_hi": "2 ग्राम प्रति लीटर पानी (0.2%)", "dosage_en": "2 g per litre of water (0.2%)",
        "freq_hi": "रोग दिखने से पहले, हर 7-10 दिन में", "freq_en": "Before disease onset, every 7-10 days", "verified": True},
    "Potato___Late_blight": {"crop": "Potato", "disease_hi": "आलू - पछेती झुलसा रोग", "disease_en": "Late Blight", "severity": "High",
        "pesticide": "बचाव: Mancozeb 75% WP | रोग होने पर: Metalaxyl+Mancozeb", "dosage_hi": "Mancozeb: 2 ग्राम/लीटर | Metalaxyl+Mancozeb: 3 ग्राम/लीटर",
        "pesticide_display": "Preventive: Mancozeb 75% WP | Curative: Metalaxyl+Mancozeb",
        "dosage_en": "Mancozeb: 2 g/litre | Metalaxyl+Mancozeb: 3 g/litre",
        "freq_hi": "बचाव के लिए 7-10 दिन में; रोग दिखने पर तुरंत बदलें", "freq_en": "Every 7-10 days preventively; switch immediately on disease onset", "verified": True},
    "Potato___healthy": {"crop": "Potato", "disease_hi": "आलू - स्वस्थ", "disease_en": "Healthy", "severity": "None",
        "pesticide": None, "dosage_hi": "आवश्यकता नहीं", "dosage_en": "Not needed", "freq_hi": "-", "freq_en": "-", "verified": None},
    "Raspberry___healthy": {"crop": "Raspberry", "disease_hi": "रसभरी - स्वस्थ", "disease_en": "Healthy", "severity": "None",
        "pesticide": None, "dosage_hi": "आवश्यकता नहीं", "dosage_en": "Not needed", "freq_hi": "-", "freq_en": "-", "verified": None},
    "Soybean___healthy": {"crop": "Soybean", "disease_hi": "सोयाबीन - स्वस्थ", "disease_en": "Healthy", "severity": "None",
        "pesticide": None, "dosage_hi": "आवश्यकता नहीं", "dosage_en": "Not needed", "freq_hi": "-", "freq_en": "-", "verified": None},
    "Squash___Powdery_mildew": {"crop": "Squash", "disease_hi": "स्क्वैश - पाउडरी मिल्ड्यू", "disease_en": "Powdery Mildew", "severity": "Medium",
        "pesticide": "Sulfur 80% WP (Wettable Sulfur)", "dosage_hi": "2-3 ग्राम प्रति लीटर पानी", "dosage_en": "2-3 g per litre of water",
        "freq_hi": "हर 10-12 दिन में", "freq_en": "Every 10-12 days", "verified": False},
    "Strawberry___Leaf_scorch": {"crop": "Strawberry", "disease_hi": "स्ट्रॉबेरी - पत्ती झुलसा", "disease_en": "Leaf Scorch", "severity": "Medium",
        "pesticide": "Captan 50% WP", "dosage_hi": "2 ग्राम प्रति लीटर पानी", "dosage_en": "2 g per litre of water",
        "freq_hi": "कटाई के बाद, हर 10-14 दिन में", "freq_en": "After harvest, every 10-14 days", "verified": False},
    "Strawberry___healthy": {"crop": "Strawberry", "disease_hi": "स्ट्रॉबेरी - स्वस्थ", "disease_en": "Healthy", "severity": "None",
        "pesticide": None, "dosage_hi": "आवश्यकता नहीं", "dosage_en": "Not needed", "freq_hi": "-", "freq_en": "-", "verified": None},
    "Tomato_Bacterial_spot": {"crop": "Tomato", "disease_hi": "टमाटर - बैक्टीरियल स्पॉट", "disease_en": "Bacterial Spot", "severity": "Medium",
        "pesticide": "Copper Oxychloride 50% WP", "dosage_hi": "2.5-3 ग्राम प्रति लीटर पानी", "dosage_en": "2.5-3 g per litre of water",
        "freq_hi": "हर 7-10 दिन में", "freq_en": "Every 7-10 days", "verified": True},
    "Tomato_Early_blight": {"crop": "Tomato", "disease_hi": "टमाटर - अगेती झुलसा रोग", "disease_en": "Early Blight", "severity": "Medium",
        "pesticide": "Mancozeb 75% WP", "dosage_hi": "2-2.5 ग्राम प्रति लीटर पानी", "dosage_en": "2-2.5 g per litre of water",
        "freq_hi": "हर 7-10 दिन में", "freq_en": "Every 7-10 days", "verified": True},
    "Tomato_Late_blight": {"crop": "Tomato", "disease_hi": "टमाटर - पछेती झुलसा रोग", "disease_en": "Late Blight", "severity": "High",
        "pesticide": "Mancozeb+Metalaxyl (या Chlorothalonil)", "pesticide_display": "Mancozeb+Metalaxyl (or Chlorothalonil)",
        "dosage_hi": "2.5-3 ग्राम प्रति लीटर पानी", "dosage_en": "2.5-3 g per litre of water",
        "freq_hi": "तुरंत छिड़काव, फिर हर 7 दिन में", "freq_en": "Spray immediately, then every 7 days", "verified": False},
    "Tomato_Leaf_Mold": {"crop": "Tomato", "disease_hi": "टमाटर - पत्ती फफूंद", "disease_en": "Leaf Mold", "severity": "Medium",
        "pesticide": "Mancozeb 75% WP", "dosage_hi": "2-2.5 ग्राम प्रति लीटर पानी", "dosage_en": "2-2.5 g per litre of water",
        "freq_hi": "हर 10 दिन में", "freq_en": "Every 10 days", "verified": False},
    "Tomato_Septoria_leaf_spot": {"crop": "Tomato", "disease_hi": "टमाटर - सेप्टोरिया पत्ती धब्बा", "disease_en": "Septoria Leaf Spot", "severity": "Medium",
        "pesticide": "Mancozeb 75% WP", "dosage_hi": "2-2.5 ग्राम प्रति लीटर पानी", "dosage_en": "2-2.5 g per litre of water",
        "freq_hi": "हर 7-10 दिन में", "freq_en": "Every 7-10 days", "verified": True},
    "Tomato_Spider_mites_Two_spotted_spider_mite": {"crop": "Tomato", "disease_hi": "टमाटर - मकड़ी कीट", "disease_en": "Spider Mites", "severity": "Medium",
        "pesticide": "नीम तेल (Neem Oil) या Dicofol 18.5% EC", "pesticide_display": "Neem Oil or Dicofol 18.5% EC",
        "dosage_hi": "नीम तेल: 5 मिली/लीटर", "dosage_en": "Neem oil: 5 ml per litre of water",
        "freq_hi": "हर 5-7 दिन में", "freq_en": "Every 5-7 days", "verified": False},
    "Tomato__Target_Spot": {"crop": "Tomato", "disease_hi": "टमाटर - टारगेट स्पॉट", "disease_en": "Target Spot", "severity": "Medium",
        "pesticide": "Mancozeb 75% WP", "dosage_hi": "2-2.5 ग्राम प्रति लीटर पानी", "dosage_en": "2-2.5 g per litre of water",
        "freq_hi": "हर 7-10 दिन में", "freq_en": "Every 7-10 days", "verified": False},
    "Tomato__Tomato_YellowLeaf__Curl_Virus": {"crop": "Tomato", "disease_hi": "टमाटर - पीली पत्ती कर्ल वायरस", "disease_en": "Yellow Leaf Curl Virus", "severity": "High",
        "pesticide": None, "dosage_hi": "कोई सीधी दवा नहीं", "dosage_en": "No direct chemical cure",
        "freq_hi": "पौधा तुरंत हटाएं; सफेद मक्खी नियंत्रण हेतु Imidacloprid प्रयोग करें", "freq_en": "Remove plant immediately; use Imidacloprid to control whitefly vector", "verified": None},
    "Tomato__Tomato_mosaic_virus": {"crop": "Tomato", "disease_hi": "टमाटर - मोज़ेक वायरस", "disease_en": "Mosaic Virus", "severity": "High",
        "pesticide": None, "dosage_hi": "कोई रासायनिक इलाज नहीं", "dosage_en": "No chemical cure",
        "freq_hi": "प्रभावित पौधे हटाएं; औजार कीटाणुरहित करें", "freq_en": "Remove affected plants; disinfect tools between plants", "verified": None},
    "Tomato_healthy": {"crop": "Tomato", "disease_hi": "टमाटर - स्वस्थ", "disease_en": "Healthy", "severity": "None",
        "pesticide": None, "dosage_hi": "आवश्यकता नहीं", "dosage_en": "Not needed", "freq_hi": "-", "freq_en": "-", "verified": None},
}

# ---------- Load Model ----------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("crop_disease_model_all.h5", compile=False, safe_mode=False)
    with open("class_indices.json", "r") as f:
        class_indices = json.load(f)
    idx_to_class = {v: k for k, v in class_indices.items()}
    return model, idx_to_class

model, idx_to_class = load_model()

# ---------- UI ----------
st.title(TEXT["title"][LANG])
st.subheader(TEXT["subtitle"][LANG])
st.caption(TEXT["tagline"][LANG])

uploaded_file = st.file_uploader(TEXT["upload_label"][LANG], type=["jpg", "jpeg", "png"])
camera_photo = st.camera_input(TEXT["camera_label"][LANG])

image_source = uploaded_file if uploaded_file else camera_photo

if image_source is not None:
    image = Image.open(image_source).convert("RGB")
    st.image(image, caption=TEXT["uploaded_caption"][LANG], use_container_width=True)

    IMG_SIZE = 224
    img_resized = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner(TEXT["analyzing"][LANG]):
        predictions = model.predict(img_array)
        predicted_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_idx] * 100
        predicted_class = idx_to_class[predicted_idx]

    info = DISEASE_DB.get(predicted_class, None)

    st.markdown("---")

    if confidence < 65:
        st.warning(TEXT["low_confidence_warning"][LANG])

    if info:
        disease_name = info["disease_hi"] if LANG == "hi" else info["disease_en"]
        severity_key = info["severity"]
        severity_display = SEVERITY_TEXT[severity_key].get(LANG, SEVERITY_TEXT[severity_key]["en"])
        crop_name = info["crop"]
    else:
        disease_name = predicted_class.replace("___", " - ").replace("_", " ")
        severity_display = "N/A"
        crop_name = "Unknown"
        info = None

    st.markdown(f"### {TEXT['result_heading'][LANG]}: **{disease_name}**")
    st.markdown(f"**{TEXT['confidence_label'][LANG]}:** {confidence:.1f}%")

    if severity_display in ["High", "उच्च"]:
        st.error(f"⚠️ {TEXT['severity_label'][LANG]}: {severity_display}")
    elif severity_display in ["Medium", "मध्यम"]:
        st.warning(f"⚠️ {TEXT['severity_label'][LANG]}: {severity_display}")
    else:
        st.success(f"✅ {TEXT['severity_label'][LANG]}: {severity_display}")

    # ---------- Pesticide + Dosage Card ----------
    if info and info["pesticide"] is None and info["severity"] != "None":
        # No-cure disease (virus/HLB/Esca)
        st.error(TEXT["no_cure_badge"][LANG])
        action = info["freq_hi"] if LANG == "hi" else info["freq_en"]
        st.markdown(f"**{action}**")
    elif info and info["severity"] != "None":
        pesticide_display = info.get("pesticide_display", info["pesticide"])
        dosage = info["dosage_hi"] if LANG == "hi" else info["dosage_en"]
        freq = info["freq_hi"] if LANG == "hi" else info["freq_en"]

        st.markdown(f"**{TEXT['pesticide_label'][LANG]}:** {pesticide_display}")
        st.markdown(f"**{TEXT['dosage_label'][LANG]}:** {dosage}")
        st.markdown(f"**{TEXT['frequency_label'][LANG]}:** {freq}")

        if info["verified"] is True:
            st.success(TEXT["verified_badge"][LANG])
        elif info["verified"] is False:
            st.warning(TEXT["estimate_badge"][LANG])

    st.markdown("---")
    st.caption(TEXT["disclaimer"][LANG])

else:
    st.info(TEXT["waiting_msg"][LANG])

# ---------- Sidebar ----------
with st.sidebar:
    st.header(TEXT["sidebar_title"][LANG])
    st.write("**SIH 2026 | Problem Statement: SIH26131**")
    st.write("14 crops, 38 disease categories.")
    st.write("Pesticide name + exact dosage + spray frequency.")
    st.write("Works fully offline once installed.")
