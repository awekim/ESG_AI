import ollama
import pandas as pd
import re
import concurrent.futures
import time 

desiredModel = 'llama3.3'

def AI_detect(transcript, business_description):
    prompt = f"""
    Business Description: {business_description}
    Firm Overview: {transcript}
**Task**  
Given the business description and overview of a company, classify it as an "AI company", "AI-related company", or "Others".  

**Evaluation Criteria (Expanded Definition)**  
- **AI company**:  
  The company either:
  - Develops AI technologies (e.g., models, algorithms, AI software, ML platforms, computer vision, NLP, etc.), **or**
  - Provides AI-powered products or services as a **core part of its business**, **or**
  - Uses AI capabilities (e.g., generative AI, predictive analytics, intelligent automation) as a **key differentiator** in its main offerings, **or**
  - Has a dedicated AI research team, AI platform, or significant AI-related R&D investment.

  *This includes companies where AI is not the only focus but is central to how the company delivers value or innovation.*

- **AI-related company**:  
  The company uses AI in a **supporting role**—e.g., for marketing optimization, logistics automation, customer service chatbots, or internal analytics—but **AI is not central** to its products or core value proposition.

- **Others**:  
  The company does not actively use, develop, or invest in AI technologies in a meaningful way. Its operations and products do not rely on AI innovation.

**Output Format**  
AI Company Status: (Choose one: "AI company", "AI-related company", or "Others")  
Explanation: (Briefly explain how the company develops or applies AI. For AI companies, emphasize how AI is central to its business model or offerings.)  
Confidence Score: (0–100%)"""
    try:
        response = ollama.chat(model=desiredModel, messages=[{"role": "user", "content": prompt}])
        result_text = response.get('message', {}).get('content', 'No response received')
        
        status_match = re.search(r"AI Company Status:\s*(.*)", result_text)
        explanation_match = re.search(r"Explanation:\s*(.*)", result_text, re.DOTALL)
        confidence_match = re.search(r"Confidence Score:\s*(\d+)", result_text)

        status = status_match.group(1).strip() if status_match else "Unknown"
        explanation = explanation_match.group(1).strip() if explanation_match else "No explanation provided"
        confidence = int(confidence_match.group(1)) if confidence_match else None
    except Exception as e:
        status = "Error"
        explanation = str(e)
        confidence = None
    return status, explanation, confidence

df_path = 'I:/Data_for_practice/ESG/final_company_list.csv'
esg_df = pd.read_csv(df_path)

esg_df = esg_df[['Targer_name', 'Target_business_description','Target_overview']].drop_duplicates()
esg_df = esg_df[esg_df['Target_overview'].notnull()].reset_index(drop=True)

targer_names = esg_df['Targer_name'].tolist()
firm_overview = esg_df['Target_overview'].tolist()
firm_business = esg_df['Target_business_description'].tolist()

results = []
for i, (name, overview, business) in enumerate(zip(targer_names, firm_overview, firm_business)):
    print(f"[{i+1}/{len(targer_names)}] 처리 중...")
    try:
        status, explanation, confidence = AI_detect(overview, business)
    except Exception as e:
        status, explanation, confidence = "Error", str(e)

    results.append({
        "Target_name": name,
        "Target_overview": overview,
        "Target_business_description": business,
        "AI Company Status": status,
        "Explanation": explanation,
        "Confidence Score": confidence
    })
    time.sleep(0.2)  
    
    df_results = pd.DataFrame(results)
    df_results.to_csv("I:/Data_for_practice/ESG/ai_target_company_classification_results.csv", index=False)


# with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
#     future_to_input = {
#         executor.submit(AI_detect, overview, business): (name, overview, business)
#         for name, overview, business in zip(targer_names, firm_overview, firm_business)
#     }
#     for i, future in enumerate(concurrent.futures.as_completed(future_to_input)):
#         name, overview, business = future_to_input[future]
#         print(f"[{i+1}/{len(future_to_input)}] 처리 중...")
#         try:
#             status, explanation, confidence = future.result()
#         except Exception as e:
#             status, explanation, confidence = "Error", str(e)

#         results.append({
#             "Targer_name": name,
#             "Target_overview": overview,
#             "Target_business_description": business,
#             "AI Company Status": status,
#             "Explanation": explanation,
#             "Confidence Score": confidence
#         })
#         time.sleep(0.2)  # Optional rate limit

df_results = pd.DataFrame(results)
df_results.to_csv("I:/Data_for_practice/ESG/ai_company_classification_results.csv", index=False)

print("AI 분류 작업 완료. 결과가 'ai_company_classification_results.csv'에 저장되었습니다.")
