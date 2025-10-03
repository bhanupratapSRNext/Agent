from typing import Literal
from langchain_openai import ChatOpenAI
import os


Intent = Literal[
    "Ecommerce_Info",          # generic ecommerce questions (go to VectorDB)
    "Ecommerce_Data",          # requires DB lookup (orders, users, inventory, etc.)
    "Feedback",
    "Support",
    "Smalltalk",
    "Abuse",
    "Unknown"
]

VALID_INTENTS = {
    "ecommerce_info",
    "ecommerce_data",
    "feedback",
    "support",
    "smalltalk",
    "abuse",
    "unknown"
}

SYSTEM_PROMPT = (
    "You are an intent classifier for an E-commerce application.\n"
    "Return exactly one of the following intents:\n\n"
    
    "- Ecommerce_Info: General questions about the e-commerce industry, "
    "trends, best practices, recommendations, guides, comparisons, or explanations. "
    "Examples: 'What is dropshipping?', 'Explain B2B vs B2C'. "
    "(These should be answered using the Vector DB.)\n\n"

    "- Ecommerce_Data: Queries that require actual data from the application's "
    "Postgres database. This includes:\n"
    "  • Orders, order status, order history\n"
    "  • Products, prices, stock, inventory, categories\n"
    "  • User account details, shipping info, payments, refunds\n"
    "  • Sales, offers, discounts, coupons, promotions, deals\n"
    "  • Any other operational/transactional details tied to the store\n"
    "(These must be answered using SQL/Postgres lookups.)\n\n"

    "- Feedback: User giving opinions, reviews, or suggestions about the system.\n"
    "- Support: User reporting an issue or asking for technical/customer help.\n"
    "- Smalltalk: Greetings, casual conversation, jokes.\n"
    "- Abuse: Toxic, rude, offensive, or harassment.\n"
    "- Unknown: Intent is unclear or does not fit any category.\n\n"

    "Return only one of the above labels, exactly as written."
)

def classify_intent(message: str) -> Intent:
    openai_key = os.getenv("OPENAI_API_KEY")
    model = "gpt-4o-mini"

    if not openai_key:
        return "UNKNOWN"

    try:
        llm = ChatOpenAI(api_key=openai_key, model=model, temperature=0)
        response = llm.invoke([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Message: {message}\nClassify the intent."},
        ])
        intent = response.content.strip().lower()

        # Match safely against valid intents (case insensitive)
        for valid_intent in VALID_INTENTS:
            if intent == valid_intent:
                return valid_intent.title() 

        return "UNKNOWN"

    # except Exception:
    except Exception as e:
        raise e
        return "UNKNOWN"


# import os
# import google.generativeai as genai
# from typing import Literal


# def classify_intent(message: str) -> str:
#     google_api_key = os.getenv("AI_API_KEY")
#     model_name = "gemini-2.5-flash"

#     if not google_api_key:
#         return "UNKNOWN"

#     try:
#         genai.configure(api_key=google_api_key)

#         # ✅ Put system_instruction here (constructor), not in generate_content()
#         model = genai.GenerativeModel(
#             model_name=model_name,
#             system_instruction=SYSTEM_PROMPT,
#             generation_config={
#                 "temperature": 0,                # deterministic for classification
#                 "max_output_tokens": 50,
#                 "response_mime_type": "text/plain",
#             },
#         )

#         # You can just pass a string; no need to build role/parts
#         resp = model.generate_content(message)

#         intent_text = (resp.text or "").strip().lower()

#         # Normalize to your canonical set (case-insensitive)
#         for valid in VALID_INTENTS:
#             if intent_text == valid.lower():
#                 # Title Case to match your original function’s return style
#                 return valid.title()

#         return "UNKNOWN"

#     except Exception as e:
#         print(f"An error occurred during intent classification: {e}")
#         return "UNKNOWN"