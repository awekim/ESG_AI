import ollama
import pandas as pd
import re
import concurrent.futures

desiredModel = 'llama3.3'

def AI_detect(transcript):
    """ 기업 설명을 받아 AI 관련성을 판별하는 함수 """
    prompt = f"""
    Transcript: {transcript}
**Task** Given a description of a company, determine whether it is AI-related.

**Criteria for Evaluation**
AI-related company: The company directly develops or heavily incorporates AI technologies (e.g., AI models, algorithms, AI-powered software, machine learning platforms, AI research, AI-enhanced products and services). This includes companies focused on AI innovation, product development, or providing AI-driven solutions in various industries (e.g., healthcare, finance, automotive, and entertainment).
Non-AI company: The company uses AI in a supplementary capacity (e.g., for marketing analytics, automation, or operational improvements) but does not have AI as a core business focus or develop AI technologies. These companies may use AI tools or services but do not engage in substantial AI innovation or research.

**Output Format**
AI Company Status: (Choose either "AI-related company" or "Non-AI company")
Explanation: (For AI-related companies, summarize how they develop or apply AI, focusing on the use of AI-driven products, services, or research. For non-AI companies, describe their primary industry and core business activities.)    """
    
    try:
        response = ollama.chat(model=desiredModel, messages=[{"role": "user", "content": prompt}])
        result_text = response.get('message', {}).get('content', 'No response received')
        
        status_match = re.search(r"AI Company Status:\s*(.*)", result_text)
        explanation_match = re.search(r"Explanation:\s*(.*)", result_text, re.DOTALL)

        status = status_match.group(1).strip() if status_match else "Unknown"
        explanation = explanation_match.group(1).strip() if explanation_match else "No explanation provided"
    except Exception as e:
        status = "Error"
        explanation = str(e)

    return status, explanation

df_path = 'I:/Data_for_practice/ESG/esg_score_deal_data.csv'
esg_df = pd.read_csv(df_path)

esg_df = esg_df[esg_df['Deal_status'].isin(['Completed', 'Completed Assumed'])]
esg_df = esg_df[['Targer_name', 'Target_overview']].drop_duplicates()
esg_df = esg_df[esg_df['Target_overview'].notnull()].reset_index(drop=True)
firm_texts = esg_df['Target_overview'].tolist()

results = []
with concurrent.futures.ThreadPoolExecutor() as executor:
    future_to_text = {executor.submit(AI_detect, text): text for text in firm_texts}
    for future in concurrent.futures.as_completed(future_to_text):
        text = future_to_text[future]
        print(text)
        try:
            status, explanation = future.result()
            results.append({
                "Company Description": text,
                "AI Company Status": status,
                "Explanation": explanation
            })
        except Exception as e:
            results.append({
                "Company Description": text,
                "AI Company Status": "Error",
                "Explanation": str(e)
            })

df_results = pd.DataFrame(results)
df_results.to_csv("I:/Data_for_practice/ESG/ai_company_classification_results.csv", index=False)

print("AI 분류 작업 완료. 결과가 'ai_company_classification_results.csv'에 저장되었습니다.")
