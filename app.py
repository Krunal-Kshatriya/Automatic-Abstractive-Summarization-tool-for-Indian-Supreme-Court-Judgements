import gradio as gr
import fitz  # PyMuPDF
import spacy
import torch
import tempfile
import re
import os
import datetime
import requests
import json
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from openai import OpenAI
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

##############################################
# Indian Kanoon API Integration
##############################################

class IndianKanoonAPI:
    """
    Class to handle interactions with the Indian Kanoon API
    """
    def __init__(self, api_token=None):
        self.base_url = "https://api.indiankanoon.org"
        self.api_token = api_token
        
    def get_headers(self):
        """Generate headers with the API token for authentication"""
        if not self.api_token:
            raise ValueError("API token is required for authentication")
        
        return {
            "Authorization": f"Token {self.api_token}",
            "Accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded"
        }
    
    def search_cases(self, query, page=0, doctypes=None, fromdate=None, todate=None, title=None, 
                     cite=None, author=None, bench=None, maxcites=None):
        """
        Search for cases matching the given criteria
        
        Parameters:
        query (str): Search terms
        page (int): Page number (starting from 0)
        doctypes (str): Type of documents to search (e.g., supremecourt, highcourts)
        fromdate (str): Start date in DD-MM-YYYY format
        todate (str): End date in DD-MM-YYYY format
        title (str): Words in the title
        cite (str): Specific citation
        author (str): Judge who authored the judgment
        bench (str): Judge who was on the bench
        maxcites (int): Maximum number of citations to return per document
        
        Returns:
        dict: Search results
        """
        print("Search cases method called with query:", query)
        url = f"{self.base_url}/search/"
        
        # Prepare data for the POST request
        data = {
            "formInput": query,
            "pagenum": str(page)  # Convert to string to ensure proper formatting
        }
        
        # Add optional parameters if provided
        if doctypes:
            data["doctypes"] = doctypes
        if fromdate:
            data["fromdate"] = fromdate
        if todate:
            data["todate"] = todate
        if title:
            data["title"] = title
        if cite:
            data["cite"] = cite
        if author:
            data["author"] = author
        if bench:
            data["bench"] = bench
        if maxcites:
            data["maxcites"] = str(maxcites)  # Convert to string
            
        try:
            # Make the POST request with the correct headers and data
            print(f"Making request to: {url}")
            print(f"With data: {data}")
            response = requests.post(
                url, 
                data=data,
                headers=self.get_headers()
            )
            
            # Check the response status
            if not response.ok:
                print(f"Error status code: {response.status_code}")
                print(f"Response content: {response.text}")
                return {"error": f"API returned status code {response.status_code}: {response.text}"}
            
            # Try to parse the response as JSON
            try:
                result = response.json()
                return result
            except ValueError:
                print(f"Invalid JSON in response: {response.text[:200]}...")
                return {"error": "Invalid JSON response from API"}
                
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
            if hasattr(e, 'response') and e.response:
                print(f"Response status: {e.response.status_code}")
                print(f"Response content: {e.response.text[:200]}...")
            return {"error": str(e)}
    
    def get_document(self, doc_id, maxcites=None, maxcitedby=None):
        """
        Get full document by ID
        
        Parameters:
        doc_id (str): Document ID
        maxcites (int): Maximum number of citations to return
        maxcitedby (int): Maximum number of cited by documents to return
        
        Returns:
        dict: Document data
        """
        url = f"{self.base_url}/doc/{doc_id}/"
        
        data = {}
        if maxcites:
            data["maxcites"] = str(maxcites)
        if maxcitedby:
            data["maxcitedby"] = str(maxcitedby)
            
        try:
            # Use POST for document retrieval
            print(f"Retrieving document with ID: {doc_id}")
            response = requests.post(url, data=data, headers=self.get_headers())
            
            if not response.ok:
                print(f"Error status code: {response.status_code}")
                print(f"Response content: {response.text}")
                return {"error": f"API returned status code {response.status_code}: {response.text}"}
            
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
            if hasattr(e, 'response') and e.response:
                print(f"Response content: {e.response.text}")
            return {"error": str(e)}
    
    def get_document_fragments(self, doc_id, query):
        """
        Get document fragments matching a query
        
        Parameters:
        doc_id (str): Document ID
        query (str): Search terms
        
        Returns:
        dict: Document fragments
        """
        url = f"{self.base_url}/docfragment/{doc_id}/"
        
        data = {
            "formInput": query
        }
            
        try:
            response = requests.post(url, data=data, headers=self.get_headers())
            if not response.ok:
                print(f"Error status code: {response.status_code}")
                print(f"Response content: {response.text}")
                return {"error": f"API returned status code {response.status_code}: {response.text}"}
            
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
            if hasattr(e, 'response') and e.response:
                print(f"Response content: {e.response.text}")
            return {"error": str(e)}
    
    def get_document_metadata(self, doc_id):
        """
        Get metadata for a document
        
        Parameters:
        doc_id (str): Document ID
        
        Returns:
        dict: Document metadata
        """
        url = f"{self.base_url}/docmeta/{doc_id}/"
            
        try:
            response = requests.post(url, headers=self.get_headers())
            if not response.ok:
                print(f"Error status code: {response.status_code}")
                print(f"Response content: {response.text}")
                return {"error": f"API returned status code {response.status_code}: {response.text}"}
            
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
            if hasattr(e, 'response') and e.response:
                print(f"Response content: {e.response.text}")
            return {"error": str(e)}


def save_judgment_as_pdf(judgment_text, title, doc_id):
    """
    Save judgment text as a PDF file
    
    Parameters:
    judgment_text (str): The text/HTML of the judgment
    title (str): Title of the judgment
    doc_id (str): Document ID
    
    Returns:
    str: Path to the saved PDF file
    """
    # Create a temporary PDF file
    pdf_file = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{doc_id}.pdf")
    pdf_path = pdf_file.name
    pdf_file.close()
    
    # Create the PDF
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Format the judgment text
    content = []
    
    # Add title
    content.append(Paragraph(f"<b>{title}</b>", styles["Title"]))
    content.append(Paragraph("<br/>", styles["Normal"]))
    
    # Remove HTML tags from judgment text (simplified approach)
    # In a real implementation, you'd want to properly parse the HTML
    import re
    clean_text = re.sub(r'<.*?>', ' ', judgment_text)
    
    # Split by paragraphs and add to content
    paragraphs = clean_text.split('\n')
    for para in paragraphs:
        if para.strip():
            content.append(Paragraph(para, styles["Normal"]))
    
    # Build the document
    doc.build(content)
    
    return pdf_path


# Local Rhetorical Role label mapping
label_mapping = {
    0: "Arguments",
    1: "Discussion by the Court",
    2: "Facts",
    3: "Issues",
    4: "Parties involved",
    5: "Precedent",
    6: "Ratio of the decision",
    7: "Ruling by lower court",
    8: "Ruling by present court",
    9: "Statute"
}

# Define claim types with their definitions for the Indian context
claim_types = {
    "Liquidated Damages Claims": "Pre-determined compensation specified in a contract that one party can claim if the other party breaches certain terms (typically related to delays or non-performance), without having to prove actual damages.",
    "Price Escalation Claims": "Claims for additional payment due to increases in the cost of materials, labor, or other resources during a contract period, often based on a formula or index specified in the contract.",
    "Interest on Delayed Payment Claims": "Claims for additional compensation based on interest accrued when payments are not made within the timeframe specified in the contract, serving as compensation for the time value of money.",
    "Scope Variation Claims": "These arise due to a change in the work scope, a revision to the specifications, or an impact to the means and methods of performing the work.",
    "Delay Claims": "These relate to schedule impacts and unanticipated project events which extend the project.",
    "Different Site Condition Claims": "These arise when some physical aspect of the project or its site differs materially from what is indicated in the contract documents.",
    "Force Majeure Claims": "These typically relate to unexpected events outside of the control of the parties, such as natural calamities.",
    "Payment Related Claims": "These arise for the non-payment or delayed payment of running bills, final bills, etc."
}

##############################################
# Global state for storing data between function calls
##############################################
class AppState:
    def __init__(self):
        self.extracted_text = None
        self.identified_claims = None
        self.claim_explanations = None
        self.annotated_text = None
        self.final_summary = None
        self.file_paths = {}
        self.rhetorical_role_counts = {}  # Store counts of each rhetorical role
        self.document_data = {"id": None, "title": None, "doc": None}  # For storing Indian Kanoon documents
        
        # Pagination state
        self.current_search_page = 0
        self.last_search_params = {}  # Store the last search parameters
        self.total_results = 0

# Initialize state
state = AppState()

##############################################
# Function to call LLM via OpenRouter API
##############################################
def call_llama_api(prompt, api_key, model_name="meta-llama/llama-4-scout:free", temperature=0.6, max_tokens=2048):
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    
    try:
        print(f"Calling API with model: {model_name}")
        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "https://legal-judgment-analyzer.app",
                "X-Title": "Legal Judgment Analysis System",
            },
            extra_body={
                "data_privacy": {
                    "prompt_tokens": "none",
                    "completion_tokens": "none"
                }
            },
            model=model_name.strip(),  # Trim any whitespace from the model name
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error with model {model_name}: {e}")
        error_message = f"API Error with model {model_name}: {str(e)}"
        # Return a more user-friendly error message
        if "404" in str(e) and "data policy" in str(e).lower():
            error_message += "\n\nThis error may be related to data privacy settings. Please try:\n1. Enabling prompt training at https://openrouter.ai/settings/privacy, or\n2. Selecting a different model that allows private prompts."
        elif "404" in str(e):
            error_message += "\n\nThe specified model may not exist or may not be available. Please check the model name for typos or try another model."
        return error_message

##############################################
# PDF Text Extraction and Annotation
##############################################
def extract_text_from_pdf(uploaded_file):
    try:
        # For Gradio file uploads, we need to handle the file path correctly
        if isinstance(uploaded_file, str):
            # If it's already a file path string
            document = fitz.open(uploaded_file)
        else:
            # If it's a file-like object or a NamedString from Gradio
            # First read the file into memory
            with open(uploaded_file.name, "rb") as file:
                file_bytes = file.read()
            
            # Then use a memory buffer to open it with PyMuPDF
            document = fitz.open(stream=file_bytes, filetype="pdf")
        
        text = "".join([page.get_text() for page in document])
        return text
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"PDF extraction error: {e}")
        print(f"Error details: {error_details}")
        return f"Failed to extract text from PDF: {e}"

def tokenize_sentences_spacy(text):
    # Load spaCy model for sentence tokenization
    nlp = spacy.load("en_core_web_lg")
    return [sent.text.strip() for sent in nlp(text).sents if sent.text.strip()]

def classify_sentences_local(sentences, model, tokenizer):
    annotated_sentences = []
    role_counts = {role: 0 for role in label_mapping.values()}
    
    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            logits = model(**inputs).logits
            predicted_label_idx = torch.argmax(logits).item()
            predicted_label = label_mapping.get(predicted_label_idx, "Unknown")
            
            # Count the occurrence of each rhetorical role
            role_counts[predicted_label] += 1
            
            annotated_sentences.append(f"[{predicted_label}] {sentence}")
    
    # Store the counts in the state
    state.rhetorical_role_counts = role_counts
    
    return annotated_sentences

##############################################
# Claim Type Classification using LLM
##############################################
def identify_claim_types(text, api_key, model_name):
    """First API call to only identify claim type names"""
    claim_type_prompt = """
    ### INSTRUCTIONS:
    You are an expert legal analyst specializing in construction claims in the Indian legal context. 
    Analyze the following legal judgment text and identify which of the following claim types are present.
    
    Claim Types to consider:
    """ + "\n".join([f"- {claim}" for claim in claim_types.keys()]) + """
    
    <TEXT TO ANALYZE>
    {text}
    </TEXT TO ANALYZE>
    
    Only include claim types that are explicitly present in the text. Format your response as follows:
    
    CLAIM TYPES IDENTIFIED:
    
    1. [Claim Type Name]
    2. [Next Claim Type Name]
    ...
    
    If no claim types are present or identifiable, state: "No specific claim types identified in this text."
    """
    
    result = call_llama_api(claim_type_prompt.format(text=text[:10000]), api_key, model_name)
    
    # Extract just the output part
    output_pattern = r"CLAIM TYPES IDENTIFIED:(.*?)(?=$)"
    output_match = re.search(output_pattern, result, re.DOTALL)
    
    if output_match:
        return output_match.group(1).strip()
    else:
        return "No claim types could be identified or the model did not provide a properly formatted response."

def explain_claim_types(api_key, model_name):
    """Second API call to get explanations for the identified claim types"""
    # Add debug output
    print(f"State contains extracted_text: {'Yes' if state.extracted_text else 'No'}")
    print(f"State contains identified_claims: {'Yes' if state.identified_claims else 'No'}")
    
    if not state.extracted_text:
        return "Error: No document text available. Please analyze a document first."
    
    if not state.identified_claims:
        return "Error: No claim types have been identified. Please analyze a document first."
    
    # Debug the content of identified_claims
    print(f"Identified claims content: {state.identified_claims}")
    
    # Extract claim names with an improved regex that handles different formats
    claim_names = []
    
    # Try different regex patterns to extract claim names
    # Pattern 1: For format like "1. [Claim Name]"
    pattern1 = r'\d+\.\s+\[(.*?)\]'
    matches1 = re.findall(pattern1, state.identified_claims)
    if matches1:
        claim_names.extend(matches1)
    
    # Pattern 2: For format like "1. Claim Name"
    if not claim_names:
        pattern2 = r'\d+\.\s+(.*?)($|\n)'
        matches2 = re.findall(pattern2, state.identified_claims)
        if matches2:
            claim_names.extend([match[0].strip() for match in matches2])
    
    # Pattern 3: For format like "Claim Name"
    if not claim_names:
        # Split by lines and filter out empty lines
        lines = [line.strip() for line in state.identified_claims.split('\n') if line.strip()]
        for line in lines:
            # Skip lines that are just numbers or don't contain alphabetic characters
            if re.match(r'^\d+\.?\s*$', line) or not any(c.isalpha() for c in line):
                continue
            # Remove leading numbers and dots
            cleaned_line = re.sub(r'^\d+\.\s*', '', line)
            # Remove any trailing parenthetical comments
            cleaned_line = re.sub(r'\s*\(.*\)\s*$', '', cleaned_line)
            claim_names.append(cleaned_line.strip())
    
    # Debug the extracted claim names
    print(f"Extracted claim names: {claim_names}")
    
    if not claim_names:
        return "No specific claim types could be extracted from the identified claims. Please check the format."
    
    explanation_prompt = """
    ### INSTRUCTIONS:
    You are an expert legal analyst specializing in construction claims in the Indian legal context. 
    I have identified the following claim types in a legal judgment:
    
    """ + "\n".join([f"- {claim}" for claim in claim_names]) + """
    
    Now, analyze the following legal judgment text and provide detailed explanations for each of these identified claim types.
    
    <TEXT TO ANALYZE>
    {text}
    </TEXT TO ANALYZE>
    
    For each claim type listed above that is present in the text, provide:
    1. The name of the claim type
    2. A detailed explanation of why this claim type is present (with reference to specific content in the text)
    3. The evidence/excerpt from the text that supports this identification
    
    Format your response as follows:
    
    CLAIM TYPE EXPLANATIONS:
    
    1. [Claim Type Name]
    - Explanation: [Your detailed explanation]
    - Evidence: "[Direct quote from text]"
    
    2. [Next Claim Type Name]
    ...
    """
    
    # Debug the prompt being sent
    print(f"Sending explanation prompt for claims: {claim_names}")
    
    result = call_llama_api(explanation_prompt.format(text=state.extracted_text[:10000]), api_key, model_name)
    
    # Extract just the output part
    output_pattern = r"CLAIM TYPE EXPLANATIONS:(.*?)(?=$)"
    output_match = re.search(output_pattern, result, re.DOTALL)
    
    if output_match:
        explanation_text = output_match.group(1).strip()
        state.claim_explanations = explanation_text
        
        # Update report with explanations
        if 'report' in state.file_paths:
            with open(state.file_paths['report']['path'], 'r') as f:
                report_content = f.read()
            
            updated_report = report_content.replace(
                "## IDENTIFIED CLAIM TYPES\n" + state.identified_claims,
                "## IDENTIFIED CLAIM TYPES\n" + state.identified_claims + "\n\n## CLAIM TYPE EXPLANATIONS\n" + explanation_text
            )
            
            with open(state.file_paths['report']['path'], 'w') as f:
                f.write(updated_report)
            
            # Save explanations to a separate file
            explanations_file = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
            explanations_file_path = explanations_file.name
            with open(explanations_file_path, 'w') as f:
                f.write(explanation_text)
                
            state.file_paths['explanations'] = {
                'path': explanations_file_path,
                'filename': f"claim_explanations.txt"
            }
        
        return explanation_text
    else:
        return "Could not generate explanations for the identified claim types. The API response did not match the expected format."

##############################################
# Custom Summary Generation Function
##############################################
def generate_custom_summary(api_key, model_name, selected_roles, summary_length):
    """Generate a customized summary based on selected rhetorical roles and desired length"""
    if not state.annotated_text:
        return "Please analyze a document first before generating a custom summary."
    
    # Debug information
    print(f"Generating custom summary with parameters:")
    print(f"- API Key: {'*' * 8}")  # Don't print the actual API key
    print(f"- Model: {model_name}")
    print(f"- Selected Roles: {selected_roles}")
    print(f"- Summary Length: {summary_length}")
    
    # Check if we have any selected roles
    if not selected_roles:
        return "Error: No rhetorical roles selected. Please select at least one role."
    
    # Filter annotated text to include only selected roles
    filtered_sentences = []
    for sentence in state.annotated_text:
        for role in selected_roles:
            if sentence.startswith(f"[{role}]"):
                filtered_sentences.append(sentence)
                break
    
    if not filtered_sentences:
        return "No sentences found for the selected rhetorical roles. Please select different roles."
    
    # Create the custom summary prompt
    custom_prompt_template = """
    ### INSTRUCTIONS:
    You are an expert legal summarizer. Carefully read the following chunk of a legal judgment that has been filtered to include only specific rhetorical roles.

    Provide a detailed summary in bulleted points focusing ONLY on the following requested section(s):
    {requested_roles}

    Guidelines for your summary:
    * Target length: Approximately {summary_length} words in total
    * Be precise: Include ONLY information explicitly stated in the provided text
    * Maintain fidelity: Do not hallucinate or add interpretations beyond what's directly present
    * Use clear attribution: When the text mentions specific parties making arguments or statements, clearly indicate who said what
    * Preserve legal terminology: Use the exact legal terms as they appear in the original text
    * If the requested section has limited content in the source text, adjust your summary length accordingly rather than adding filler or interpretation
    * For statutory sections: Focus on the precise language describing provisions, requirements, and timeframes without elaboration
    * For precedent sections: Note case names, citations, and the specific principles established

    Format your summary with:
    * Hierarchical bulleted structure 
    * Main points as primary bullets
    * Supporting details as secondary bullets
    * Bold text for critical elements (like time limits, key requirements, or pivotal holdings)

    <TEXT TO SUMMARIZE>
    {filtered_text}
    </TEXT TO SUMMARIZE>
        
    ### {requested_roles} SUMMARY:    """
    
    # Format the roles for the prompt
    roles_text = "\n".join([f"- {role}" for role in selected_roles])
    
    # Combine filtered text
    filtered_text = " ".join(filtered_sentences)
    
    # Format the prompt
    custom_prompt = custom_prompt_template.format(
        requested_roles=roles_text,
        summary_length=summary_length,
        filtered_text=filtered_text
    )
    
    # Call the API with the custom prompt
    custom_summary = call_llama_api(custom_prompt, api_key, model_name)
    
    # Save the custom summary to a file
    if hasattr(state, 'file_paths') and 'report' in state.file_paths:
        filename = os.path.basename(state.file_paths['report']['path'])
        base_name = os.path.splitext(filename)[0]
        
        custom_summary_file = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
        custom_summary_file_path = custom_summary_file.name
        
        with open(custom_summary_file_path, 'w') as f:
            f.write(f"# CUSTOM SUMMARY\n")
            f.write(f"## Selected roles: {', '.join(selected_roles)}\n")
            f.write(f"## Target length: {summary_length} words\n\n")
            f.write(custom_summary)
        
        state.file_paths['custom_summary'] = {
            'path': custom_summary_file_path,
            'filename': f"{base_name}_custom_summary.txt"
        }
    
    return custom_summary

##############################################
# Main Analysis Function
##############################################
def analyze_document(pdf_file, api_key, model_name, model_path=None):
    if not pdf_file or not api_key:
        return "Please upload a PDF file and provide an API key.", "", "", "", "", {}
    
    # Use default path if not provided
    if not model_path:
        model_path = "./trained_model"
    
    try:
        # Step 1: Load the models
        try:
            local_model = AutoModelForSequenceClassification.from_pretrained(model_path)
            local_tokenizer = AutoTokenizer.from_pretrained(model_path)
            print(f"Successfully loaded model from {model_path}")
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
            return f"Error loading model: {str(e)}", "", "", "", "", {}
        
        # Step 2: Extract text from PDF
        extracted_text = extract_text_from_pdf(pdf_file)
        state.extracted_text = extracted_text
        
        # Step 3: Tokenize and classify sentences
        sentences = tokenize_sentences_spacy(extracted_text)
        annotated_text = classify_sentences_local(sentences, local_model, local_tokenizer)
        state.annotated_text = annotated_text
        
        # Save annotated text to file
        annotated_file = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
        annotated_file_path = annotated_file.name
        with open(annotated_file_path, 'w') as f:
            f.write("\n".join(annotated_text))
        
        filename = os.path.basename(pdf_file)
        state.file_paths['annotated'] = {
            'path': annotated_file_path,
            'filename': f"{os.path.splitext(filename)[0]}_annotated_text.txt"
        }
        
        # Step 4: Identify claim types using the selected model
        identified_claims = identify_claim_types(extracted_text, api_key, model_name)
        state.identified_claims = identified_claims
        
        # Save claim types to file
        claim_file = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
        claim_file_path = claim_file.name
        with open(claim_file_path, 'w') as f:
            f.write(identified_claims)
            
        state.file_paths['claims'] = {
            'path': claim_file_path,
            'filename': f"{os.path.splitext(filename)[0]}_claim_types.txt"
        }
        
        # Step 5: Generate summary using the selected model
        stuff_prompt_template = """
        ### INSTRUCTIONS:
        You are an expert legal summarizer. Carefully read the following chunk of a legal judgment annotated with rhetorical roles.
        Provide a detailed summary in bulleted points covering all present sections:
        - Parties involved
        - Facts
        - Issues
        - Arguments
        - Rulings by lower court
        - Statute
        - Discussion by the court
        - Precedent
        - Ratio of the decision
        - Ruling by present court

        ### EXAMPLE SUMMARY FORMAT:
        Parties involved: 
        - ABC Ltd. vs XYZ Ltd.
        Facts: 
        - The dispute arose over a breach of contract regarding construction services.
        Issues: 
        - Whether the penalty clause was enforceable.
        Arguments: 
        - ABC argued the clause was excessive; XYZ cited prior case law.
        Ruling by lower court: 
        - Favored ABC citing unfair penalties.
        Statute: 
        - Indian Contract Act, Section 74.
        Discussion by the court: 
        - Focused on whether penalties were compensatory or punitive.
        Precedent: 
        - Referred to Fateh Chand v. Balkishan Das.
        Ratio of the decision: 
        - Penalty must be a genuine pre-estimate of loss.
        Ruling by present court: 
        - The court ruled in favor of XYZ, limiting damages to actual loss.

        
        <TEXT TO SUMMARIZE>
        {full_text}
        </TEXT TO SUMMARIZE>

        
        ### DETAILED SUMMARY:

        """
        
        full_text = " ".join(annotated_text)  # Combine annotated text
        stuff_prompt = stuff_prompt_template.format(full_text=full_text)
        final_summary = call_llama_api(stuff_prompt, api_key, model_name)
        state.final_summary = final_summary
        
        # Save summary to file
        summary_file = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
        summary_file_path = summary_file.name
        with open(summary_file_path, 'w') as f:
            f.write(final_summary)
            
        state.file_paths['summary'] = {
            'path': summary_file_path,
            'filename': f"{os.path.splitext(filename)[0]}_summary.txt"
        }
        
        # Step 6: Create consolidated report
        current_datetime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        consolidated_report = f"""# LEGAL JUDGMENT ANALYSIS REPORT
## Document: {filename}
## Date: {current_datetime}
## Model used: {model_name}

## IDENTIFIED CLAIM TYPES
{identified_claims}

## COMPREHENSIVE SUMMARY
{final_summary}
"""
        
        consolidated_file = tempfile.NamedTemporaryFile(delete=False, suffix=".md")
        consolidated_file_path = consolidated_file.name
        with open(consolidated_file_path, 'w') as f:
            f.write(consolidated_report)
            
        state.file_paths['report'] = {
            'path': consolidated_file_path,
            'filename': f"{os.path.splitext(filename)[0]}_full_report.md"
        }
        
        # Return a message for the download status and the role counts
        download_info = "Analysis completed. Files are ready for download. Go to the Downloads tab and click 'Show Files for Download'."
        
        return "\n".join(annotated_text), identified_claims, final_summary, "", download_info, state.rhetorical_role_counts
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Analysis error: {e}")
        print(f"Error details: {error_details}")
        return f"An error occurred during analysis: {str(e)}", "", "", "", "", {}

##############################################
# File download helper functions
##############################################
def get_file_contents():
    """Create text content for download buttons from files in state.file_paths"""
    file_contents = {}
    
    for key, file_info in state.file_paths.items():
        if os.path.exists(file_info['path']):
            try:
                with open(file_info['path'], 'r', encoding='utf-8') as f:
                    file_content = f.read()
                    file_contents[file_info['filename']] = file_content
            except Exception as e:
                print(f"Error reading file {file_info['path']}: {e}")
    
    # Print debug info
    print(f"Found {len(file_contents)} files to download")
    for filename in file_contents.keys():
        print(f"  - {filename}")
    
    # Return the dictionary of filenames and contents
    return file_contents

# Function to update the dropdown when refresh is clicked
def update_files_dropdown():
    file_contents = get_file_contents()
    if file_contents:
        return gr.update(choices=list(file_contents.keys()), value=next(iter(file_contents.keys()), None))
    else:
        return gr.update(choices=[], value=None)

# Function to show the selected file content
def show_file_content(selected_file):
    if not selected_file:
        return ""
    
    file_contents = get_file_contents()
    if selected_file in file_contents:
        return file_contents[selected_file]
    return ""

##############################################
# Create search tab for Indian Kanoon integration
##############################################
def create_search_tab():
    """Create the court case search tab for the Gradio interface"""
    
    with gr.Tab("Court Case Search"):
        gr.Markdown("## Search Indian Kanoon")
        
        with gr.Row():
            with gr.Column(scale=1):
                api_token = gr.Textbox(label="Indian Kanoon API Token", type="password")
                
                with gr.Row():
                    search_query = gr.Textbox(label="Search Query", placeholder="Enter keywords (e.g., 'liquidated damages construction delay')")
                
                with gr.Accordion("Advanced Search Options", open=False):
                    doc_type = gr.Dropdown(
                        label="Document Type",
                        choices=[
                            "All Documents",
                            "Supreme Court",
                            "High Courts",
                            "District Courts",
                            "Tribunals"
                        ],
                        value="All Documents"
                    )
                    
                    with gr.Row():
                        from_date = gr.Textbox(label="From Date (DD-MM-YYYY)", placeholder="e.g., 01-01-2015")
                        to_date = gr.Textbox(label="To Date (DD-MM-YYYY)", placeholder="e.g., 31-12-2022")
                    
                    with gr.Row():
                        title_filter = gr.Textbox(label="Words in Title", placeholder="Filter by title words")
                        citation_filter = gr.Textbox(label="Citation", placeholder="e.g., 2015 AIR 123")
                    
                    with gr.Row():
                        author_filter = gr.Textbox(label="Author Judge", placeholder="Judge who wrote the judgment")
                        bench_filter = gr.Textbox(label="Bench Judge", placeholder="Judge who was on the bench")
                
                search_button = gr.Button("Search Cases", variant="primary")
            
            with gr.Column(scale=2):
                search_results_info = gr.Textbox(label="Search Results Summary", lines=1)
                search_results = gr.Dataframe(
                    headers=["ID", "Title", "Source", "Date", "Size", "Snippet"],
                    label="Search Results",
                    row_count=10,
                    col_count=(6, "fixed"),
                    interactive=False
                )
                
                # Add pagination controls
                with gr.Row():
                    prev_page_button = gr.Button("Previous Page", variant="secondary")
                    current_page_display = gr.Textbox(label="Page", value="1", interactive=False)
                    next_page_button = gr.Button("Next Page", variant="secondary")
        
        with gr.Row():
            with gr.Column(scale=1):
                selected_case_id = gr.Textbox(label="Selected Document ID", placeholder="Click on a row to select a case")
                view_doc_button = gr.Button("View Document")
            
            with gr.Column(scale=2):
                document_title = gr.Textbox(label="Document Title")
                document_preview = gr.HTML(label="Document Preview")
        
        with gr.Row():
            analyze_button = gr.Button("Download & Analyze Selected Case", variant="primary")
            download_status = gr.Textbox(label="Download Status", visible=True)
    
    return {
        "api_token": api_token,
        "search_query": search_query,
        "doc_type": doc_type,
        "from_date": from_date,
        "to_date": to_date,
        "title_filter": title_filter,
        "citation_filter": citation_filter,
        "author_filter": author_filter,
        "bench_filter": bench_filter,
        "search_button": search_button,
        "search_results_info": search_results_info,
        "search_results": search_results,
        "selected_case_id": selected_case_id,
        "view_doc_button": view_doc_button,
        "document_title": document_title,
        "document_preview": document_preview,
        "analyze_button": analyze_button,
        "download_status": download_status,
        "prev_page_button": prev_page_button,
        "current_page_display": current_page_display,
        "next_page_button": next_page_button
    }

# Function to handle pagination and search
def perform_search(api_token, query, doc_type, from_date, to_date, title, cite, author, bench, page=0):
    if not api_token or not query:
        return "Please provide an API token and search query", None, "1"
    
    # Initialize API client
    api = IndianKanoonAPI(api_token)
    
    # Map document type
    doctypes = map_doc_type(doc_type)
    
    # Store search parameters for pagination
    state.last_search_params = {
        "query": query,
        "doctypes": doctypes,
        "fromdate": from_date if from_date else None,
        "todate": to_date if to_date else None,
        "title": title if title else None,
        "cite": cite if cite else None,
        "author": author if author else None,
        "bench": bench if bench else None,
        "maxcites": 5
    }
    
    # Update current page in state
    state.current_search_page = page
    
    # Make the search request
    result = api.search_cases(
        query=query,
        page=page,
        doctypes=doctypes,
        fromdate=from_date if from_date else None,
        todate=to_date if to_date else None,
        title=title if title else None,
        cite=cite if cite else None,
        author=author if author else None,
        bench=bench if bench else None,
        maxcites=5  # Get a few citations for each result
    )
    
    if "error" in result:
        return f"Error: {result['error']}", None, "1"
    
    # Format for display
    docs = result.get("docs", [])
    total_found_str = result.get("found", "0")
    
    # Extract the actual total count from the formatted string
    # It might be in format "1 - 10 of 9638" or just a number
    print(f"Found value: {total_found_str}, type: {type(total_found_str)}")
    
    try:
        if isinstance(total_found_str, str) and " of " in total_found_str:
            # Extract the number after "of"
            total_found = int(total_found_str.split(" of ")[-1].strip().replace(",", ""))
        else:
            # It's already a number or a simple numeric string
            total_found = int(str(total_found_str).replace(",", ""))
    except (ValueError, TypeError, AttributeError):
        print(f"Could not parse total results count from '{total_found_str}', using length of docs")
        total_found = len(docs)
    
    # Store the total results count in state
    state.total_results = total_found
    
    if not docs:
        return f"No results found for query: {query}", None, "1"
    
    # Prepare data for the table
    table_data = []
    for doc in docs:
        # Extract document ID safely
        tid = doc.get("tid", "")
        if isinstance(tid, int):
            doc_id = str(tid)
        elif isinstance(tid, str) and "/" in tid:
            doc_id = tid.split("/")[-1]
        else:
            doc_id = str(tid)
        
        # Format the data
        table_data.append([
            doc_id,
            doc.get("title", ""),
            doc.get("docsource", ""),
            doc.get("posted_date", ""),
            doc.get("docsize", 0),
            doc.get("headline", "")
        ])
    
    # Calculate page range for display
    start_result = page * 10 + 1
    end_result = start_result + len(docs) - 1
    current_page = str(page + 1)  # Convert to 1-based for display
    
    return f"Found {total_found} results for query: {query}. Showing {start_result} - {end_result}", table_data, current_page

def go_to_previous_page(api_token):
    """Go to the previous page of search results"""
    # Safety check
    if not hasattr(state, 'current_search_page') or not hasattr(state, 'last_search_params'):
        return "No previous search to navigate", None, "1"
    
    # Decrement page if possible
    if state.current_search_page > 0:
        new_page = state.current_search_page - 1
    else:
        new_page = 0  # Stay on first page
    
    # Re-execute search with new page
    return perform_search(
        api_token=api_token,
        query=state.last_search_params.get("query", ""),
        doc_type=state.last_search_params.get("doctypes", "All Documents"),
        from_date=state.last_search_params.get("fromdate", None),
        to_date=state.last_search_params.get("todate", None),
        title=state.last_search_params.get("title", None),
        cite=state.last_search_params.get("cite", None),
        author=state.last_search_params.get("author", None),
        bench=state.last_search_params.get("bench", None),
        page=new_page
    )

def go_to_next_page(api_token):
    """Go to the next page of search results"""
    # Safety check
    if not hasattr(state, 'current_search_page') or not hasattr(state, 'last_search_params'):
        return "No previous search to navigate", None, "1"
    
    # Make sure total_results is an integer
    try:
        total_results = int(state.total_results)
    except (ValueError, TypeError):
        total_results = 0
    
    # Calculate total pages (ceiling division)
    total_pages = (total_results + 9) // 10
    
    # Increment page if possible
    if state.current_search_page < total_pages - 1:
        new_page = state.current_search_page + 1
    else:
        new_page = state.current_search_page  # Stay on last page
    
    # Re-execute search with new page
    return perform_search(
        api_token=api_token,
        query=state.last_search_params.get("query", ""),
        doc_type=state.last_search_params.get("doctypes", "All Documents"),
        from_date=state.last_search_params.get("fromdate", None),
        to_date=state.last_search_params.get("todate", None),
        title=state.last_search_params.get("title", None),
        cite=state.last_search_params.get("cite", None),
        author=state.last_search_params.get("author", None),
        bench=state.last_search_params.get("bench", None),
        page=new_page
    )

def map_doc_type(doc_type):
    """Map the document type from UI selection to API parameter"""
    mapping = {
        "All Documents": None,
        "Supreme Court": "supremecourt",
        "High Courts": "highcourts",
        "District Courts": "delhidc",
        "Tribunals": "tribunals"
    }
    return mapping.get(doc_type)

def integrate_search_with_main_app(pdf_file_component, api_key_component, model_name_component, analyze_button_component):
    """
    Integrate the search tab with the main application
    
    Parameters:
    pdf_file_component: The file upload component from the main app
    api_key_component: The API key component from the main app
    model_name_component: The model name component from the main app
    analyze_button_component: The analyze button from the main app
    """
    # Create the search components
    search_components = create_search_tab()
    
    # Connect search button
    search_components["search_button"].click(
        fn=perform_search,
        inputs=[
            search_components["api_token"],
            search_components["search_query"],
            search_components["doc_type"],
            search_components["from_date"],
            search_components["to_date"],
            search_components["title_filter"],
            search_components["citation_filter"],
            search_components["author_filter"],
            search_components["bench_filter"],
        ],
        outputs=[
            search_components["search_results_info"],
            search_components["search_results"],
            search_components["current_page_display"]
        ]
    )
    
    # Connect pagination buttons
    search_components["prev_page_button"].click(
        fn=go_to_previous_page,
        inputs=[search_components["api_token"]],
        outputs=[
            search_components["search_results_info"],
            search_components["search_results"],
            search_components["current_page_display"]
        ]
    )
    
    search_components["next_page_button"].click(
        fn=go_to_next_page,
        inputs=[search_components["api_token"]],
        outputs=[
            search_components["search_results_info"],
            search_components["search_results"],
            search_components["current_page_display"]
        ]
    )
    
    # Handle row selection in the results table
    def select_case(evt: gr.SelectData, results):
        try:
            selected_row = evt.index[0]
            # Avoid the DataFrame truth value ambiguity
            if results is not None and isinstance(results, list) and len(results) > selected_row:
                # Get the document ID from the first column
                doc_id = results[selected_row][0]
                return doc_id
        except Exception as e:
            print(f"Error selecting case: {e}")
            return ""
        return ""
    
    search_components["search_results"].select(
        fn=select_case,
        inputs=[search_components["search_results"]],
        outputs=[search_components["selected_case_id"]]
    )
    
    # View document button
    def view_document(api_token, doc_id):
        if not api_token or not doc_id:
            return "Please select a case and provide an API token", ""
            
        # Initialize API client
        api = IndianKanoonAPI(api_token)
        
        # Get document
        result = api.get_document(doc_id)
        
        if "error" in result:
            return f"Error: {result['error']}", ""
            
        # Store document data for later use
        state.document_data["id"] = doc_id
        state.document_data["title"] = result.get("title", "Untitled Document")
        state.document_data["doc"] = result.get("doc", "")
        
        # Return preview
        return result.get("title", ""), result.get("doc", "")
    
    search_components["view_doc_button"].click(
        fn=view_document,
        inputs=[
            search_components["api_token"],
            search_components["selected_case_id"]
        ],
        outputs=[
            search_components["document_title"],
            search_components["document_preview"]
        ]
    )
    
    # Analyze document button
    def download_and_analyze(api_token, doc_id, api_key_value, model_name_value):
        if not api_token or not doc_id:
            return "Please select a case and provide an API token", None
        
        if not api_key_value:
            return "Please provide an OpenRouter API key for analysis", None
            
        # Check if we've already loaded the document
        if state.document_data["id"] != doc_id or not state.document_data["doc"]:
            # Initialize API client
            api = IndianKanoonAPI(api_token)
            
            # Get document
            result = api.get_document(doc_id)
            
            if "error" in result:
                return f"Error: {result['error']}", None
                
            state.document_data["id"] = doc_id
            state.document_data["title"] = result.get("title", "Untitled Document")
            state.document_data["doc"] = result.get("doc", "")
        
        # Save as PDF
        try:
            pdf_path = save_judgment_as_pdf(
                state.document_data["doc"], 
                state.document_data["title"], 
                doc_id
            )
            
            return f"Downloaded document: {state.document_data['title']}", pdf_path
        except Exception as e:
            print(f"Error saving PDF: {e}")
            return f"Error converting document to PDF: {str(e)}", None
    
    # Connect download button to prepare the document
    search_components["analyze_button"].click(
        fn=download_and_analyze,
        inputs=[
            search_components["api_token"],
            search_components["selected_case_id"],
            api_key_component,
            model_name_component
        ],
        outputs=[
            search_components["download_status"],
            pdf_file_component
        ]
    )
    
    # Add tooltip function to automatically trigger analysis after download
    def trigger_analysis():
        return gr.update(visible=True)
    
    search_components["analyze_button"].click(
        fn=trigger_analysis,
        inputs=[],
        outputs=[analyze_button_component]
    )
    
    return search_components
    
##############################################
# Helper function to update custom summary controls
##############################################
def update_role_checkboxes(role_counts):
    """Update the rhetorical role checkboxes with counts"""
    checkbox_updates = {}
    for role, count in role_counts.items():
        checkbox_updates[f"{role}_checkbox"] = gr.update(label=f"{role} ({count} sentences)")
    
    return checkbox_updates

##############################################
# Gradio Interface
##############################################
with gr.Blocks(title="Legal Judgment Analysis System") as demo:
    gr.Markdown("# Legal Judgment Analysis System")
    gr.Markdown("## Rhetorical Role Annotation & Claim Type Classification with Court Case Search")
    
    with gr.Tabs() as tabs:
        # Original analysis tab
        with gr.Tab("Document Analysis"):
            with gr.Row():
                with gr.Column(scale=1):
                    api_key = gr.Textbox(label="Enter your OpenRouter API Key", type="password")
                    model_name = gr.Textbox(
                        label="Enter LLM Model Name", 
                        value="meta-llama/llama-4-scout:free",
                        placeholder="e.g., meta-llama/llama-4-scout:free, anthropic/claude-3-opus:beta"
                    )
                    pdf_file = gr.File(label="Upload PDF File", file_types=[".pdf"])
                    model_path = gr.Textbox(
                        label="Path to trained model (optional)", 
                        value="./trained_model",
                        visible=True
                    )
                    analyze_button = gr.Button("Analyze Document", elem_id="analyze-button")
                
                with gr.Column(scale=1):
                    with gr.Accordion("About Claim Types", open=False):
                        gr.Markdown("### Claim Types in Indian Construction Law")
                        claim_types_md = "\n\n".join([f"**{claim}**: {definition}" for claim, definition in claim_types.items()])
                        gr.Markdown(claim_types_md)
                    
                    with gr.Accordion("Available LLM Models", open=False):
                        gr.Markdown("### Common Models on OpenRouter")
                        gr.Markdown("""
                        Here are some common models you can use:
                        
                        - **meta-llama/llama-4-scout:free** - Free tier of Llama 4 Scout
                        - **anthropic/claude-3-opus:beta** - Claude 3 Opus model
                        - **anthropic/claude-3-5-sonnet:beta** - Claude 3.5 Sonnet model
                        - **mistralai/mistral-large:latest** - Latest Mistral Large model
                        - **google/gemma-7b:latest** - Latest Gemma 7B model
                        
                        Check OpenRouter documentation for more available models.
                        """)
            
            with gr.Tabs() as result_tabs:
                with gr.TabItem("Annotated Text"):
                    annotated_output = gr.Textbox(label="Labeled Sentences (Rhetorical Roles)", lines=15)
                
                with gr.TabItem("Claim Types"):
                    claims_output = gr.Textbox(label="Identified Claim Types", lines=10)
                    explain_button = gr.Button("Explain the Identified Claim Types")
                    explanations_output = gr.Textbox(label="Claim Type Explanations", lines=15, visible=True)
                
                with gr.TabItem("Summary"):
                    summary_output = gr.Textbox(label="Complete Legal Judgment Summary", lines=15)
                
                with gr.TabItem("Custom Summary"):
                    gr.Markdown("### Create a Custom Summary")
                    gr.Markdown("Select which rhetorical roles to include in your summary and specify the desired length.")
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            # Create checkboxes for each rhetorical role
                            role_checkboxes = {}
                            for role in label_mapping.values():
                                role_checkboxes[f"{role}_checkbox"] = gr.Checkbox(
                                    label=f"{role}", 
                                    value=True,
                                    interactive=True
                                )
                            
                            # Add a "Select All" checkbox
                            select_all_checkbox = gr.Checkbox(
                                label="Select All Roles", 
                                value=True,
                                interactive=True
                            )
                        
                        with gr.Column(scale=1):
                            summary_length = gr.Slider(
                                label="Summary Length (words)", 
                                minimum=100, 
                                maximum=500, 
                                value=250, 
                                step=50
                            )
                            custom_summary_button = gr.Button("Generate Custom Summary", variant="primary")
                            custom_summary_output = gr.Textbox(label="Custom Summary", lines=15)
                    
                with gr.TabItem("Downloads"):
                    download_info = gr.Textbox(label="Download Status", lines=1)
                    refresh_downloads_btn = gr.Button("Show Files for Download")
                    files_dropdown = gr.Dropdown(label="Select a file to download", choices=[], interactive=True)
                    file_content = gr.TextArea(label="File Content (copy this text to save the file)", lines=20, interactive=False)
        
        # Integrate the search functionality
        # The function will create and return the search components
        search_components = integrate_search_with_main_app(
            pdf_file,
            api_key,
            model_name,
            analyze_button
        )
    
    # Hidden component to store role counts
    role_counts_state = gr.State({})
    
    # Function to handle select all checkbox
    def toggle_all_checkboxes(select_all):
        return [gr.update(value=select_all) for _ in role_checkboxes]
    
    # Connect functions to buttons
    analyze_button.click(
        fn=analyze_document,
        inputs=[pdf_file, api_key, model_name, model_path],
        outputs=[annotated_output, claims_output, summary_output, explanations_output, download_info, role_counts_state]
    ).then(
        fn=update_role_checkboxes,
        inputs=[role_counts_state],
        outputs=[checkbox for checkbox in role_checkboxes.values()]
    )
    
    # Connect select all checkbox
    select_all_checkbox.change(
        fn=toggle_all_checkboxes,
        inputs=[select_all_checkbox],
        outputs=[checkbox for checkbox in role_checkboxes.values()]
    )
    
    explain_button.click(
        fn=explain_claim_types,
        inputs=[api_key, model_name],
        outputs=[explanations_output]
    )
    
    # Handle custom summary generation
    def process_custom_summary(api_key_val, model_name_val, summary_length_val, *checkbox_values):
        # Get the roles that are selected (checkbox is True)
        selected_roles = []
        for role, is_selected in zip(label_mapping.values(), checkbox_values):
            if is_selected:
                selected_roles.append(role)
        
        # Call the generate_custom_summary function with processed inputs
        return generate_custom_summary(api_key_val, model_name_val, selected_roles, summary_length_val)
    
    custom_summary_button.click(
        fn=process_custom_summary,
        inputs=[api_key, model_name, summary_length] + list(role_checkboxes.values()),
        outputs=[custom_summary_output]
    )
    
    # Connect the refresh button to update the dropdown
    refresh_downloads_btn.click(
        fn=update_files_dropdown,
        inputs=[],
        outputs=[files_dropdown]
    )
    
    # Connect the dropdown to show the selected file content
    files_dropdown.change(
        fn=show_file_content,
        inputs=[files_dropdown],
        outputs=[file_content]
    )
    
# Handle cleanup of temporary files
import atexit

def cleanup_temp_files():
    if hasattr(state, 'file_paths'):
        for key in state.file_paths:
            try:
                os.remove(state.file_paths[key]['path'])
            except:
                pass

atexit.register(cleanup_temp_files)

# Launch the app
if __name__ == "__main__":
    demo.launch()