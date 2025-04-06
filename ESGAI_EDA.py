import ollama
import pandas as pd
import re
import concurrent.futures

desiredModel = 'llama3.3'

def AI_detect(transcript, business_description):
    prompt = f"""
    Business Description: {business_description}
    Firm Overview: {transcript}
**Task** Given the business description and overview of a company, determine whether it is AI-related.

**Criteria for Evaluation**
AI-related company: The company directly develops or heavily incorporates AI technologies (e.g., AI models, algorithms, AI-powered software, machine learning platforms, AI research, AI-enhanced products and services). This includes companies focused on AI innovation, product development, or providing AI-driven solutions in various industries (e.g., healthcare, finance, automotive, and entertainment).
Non-AI company: The company uses AI in a supplementary capacity (e.g., for marketing analytics, automation, or operational improvements) but does not have AI as a core business focus or develop AI technologies. These companies may use AI tools or services but do not engage in substantial AI innovation or research.

**Output Format**
AI Company Status: (Choose either "AI-related company" or "Non-AI company")
Explanation: (For AI-related companies, summarize how they develop or apply AI, focusing on the use of AI-driven products, services, or research. For non-AI companies, describe their primary industry and core business activities.)    
Confidence Score: (Provide a percentage score from 0 to 100, representing your confidence in the classification.)"""
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
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    future_to_input = {
        executor.submit(AI_detect, overview, business): (name, overview, business)
        for name, overview, business in zip(targer_names, firm_overview, firm_business)
    }

    for i, future in enumerate(concurrent.futures.as_completed(future_to_input)):
        name, overview, business = future_to_input[future]
        print(f"[{i+1}/{len(future_to_input)}] 처리 중...")
        try:
            status, explanation, confidence = future.result()
        except Exception as e:
            status, explanation, confidence = "Error", str(e)

        results.append({
            "Targer_name": name,
            "Target_overview": overview,
            "Target_business_description": business,
            "AI Company Status": status,
            "Explanation": explanation,
            "Confidence Score": confidence
        })
        time.sleep(0.2)  # Optional rate limit

df_results = pd.DataFrame(results)
df_results.to_csv("I:/Data_for_practice/ESG/ai_company_classification_results.csv", index=False)

print("AI 분류 작업 완료. 결과가 'ai_company_classification_results.csv'에 저장되었습니다.")
