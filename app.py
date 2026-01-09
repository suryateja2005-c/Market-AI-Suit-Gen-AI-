from flask import Flask, render_template, request, jsonify
import requests
import json
import time

app = Flask(__name__)

# Configuration
HF_API_KEY = "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" #enter your Hugging Face API key here
MODEL_ID = "ibm-granite/granite-3.0-8b-instruct"  # Using this exact model
HF_API_URL = "https://router.huggingface.co/chat/completions"

print(f"\n{'='*70}")
print(f"🎓 GenAI Curriculum Generator - Starting")
print(f"{'='*70}")
print(f"✅ API Key: {HF_API_KEY[:25]}...")
print(f"✅ Model: {MODEL_ID}")
print(f"✅ API Endpoint: {HF_API_URL}")
print(f"{'='*70}\n")

def generate_curriculum(skill, level, semesters, weekly_hours, industry_focus):
    """Generate curriculum using IBM Granite 3.0 8B Instruct via Hugging Face Router API"""
    
    print(f"\n{'='*70}")
    print(f"🔄 CURRICULUM GENERATION REQUEST")
    print(f"{'='*70}")
    print(f"📚 Skill: {skill}")
    print(f"🎓 Level: {level}")
    print(f"⏱️  Semesters: {semesters}")
    print(f"⏰ Weekly Hours: {weekly_hours or '20-25'}")
    print(f"🏢 Industry Focus: {industry_focus or 'General Tech'}")
    print(f"{'='*70}\n")
    
    # Optimized prompt for IBM Granite model
    prompt = f"""Create a detailed curriculum for {skill} at {level} level.

SPECIFICATIONS:
- Total Semesters: {semesters}
- Hours per week: {weekly_hours if weekly_hours else '20-25'}
- Industry Focus: {industry_focus if industry_focus else 'General Tech'}
- Progressive difficulty: Beginner → Intermediate → Advanced

RETURN ONLY THIS EXACT JSON STRUCTURE (no other text):

{{
  "skill": "{skill}",
  "level": "{level}",
  "totalSemesters": {semesters},
  "weeklyHours": "{weekly_hours if weekly_hours else '20-25'}",
  "industryFocus": "{industry_focus if industry_focus else 'General Tech'}",
  "semesters": [
    {{
      "semesterNumber": 1,
      "title": "Foundation Concepts",
      "duration": "4 months",
      "subjects": [
        {{
          "name": "Core Subject 1",
          "creditHours": 4,
          "hoursPerWeek": 4,
          "units": [
            {{
              "unitNumber": 1,
              "unitName": "Basics",
              "topics": ["Introduction", "Fundamentals"],
              "learningOutcomes": ["Understand core concepts", "Apply basic principles"],
              "hoursPerWeek": 2
            }}
          ],
          "tools": ["Tool 1", "Tool 2"],
          "miniProject": {{
            "title": "Beginner Project",
            "description": "Simple implementation project",
            "duration": "2 weeks",
            "deliverables": ["Code", "Documentation"]
          }},
          "assessment": {{"theory": 40, "practical": 30, "project": 30}}
        }}
      ]
    }},
    {{
      "semesterNumber": 2,
      "title": "Intermediate Development",
      "duration": "4 months",
      "subjects": [
        {{
          "name": "Core Subject 2",
          "creditHours": 4,
          "hoursPerWeek": 4,
          "units": [
            {{
              "unitNumber": 1,
              "unitName": "Advanced Topics",
              "topics": ["Complex concepts", "Best practices"],
              "learningOutcomes": ["Master advanced techniques", "Implement solutions"],
              "hoursPerWeek": 2
            }}
          ],
          "tools": ["Tool 1", "Tool 2"],
          "miniProject": {{
            "title": "Intermediate Project",
            "description": "More complex implementation",
            "duration": "2 weeks",
            "deliverables": ["Code", "Documentation"]
          }},
          "assessment": {{"theory": 40, "practical": 30, "project": 30}}
        }}
      ]
    }}
  ],
  "capstoneProject": {{
    "title": "Capstone: Real-world {skill} Application",
    "description": "Build a complete {skill} project from scratch with industry standards",
    "duration": "8 weeks",
    "requirements": ["Planning", "Implementation", "Testing", "Documentation"]
  }},
  "internshipReadiness": [
    "Advanced {skill} skills",
    "Industry tools and frameworks",
    "Software development best practices",
    "Team collaboration and communication",
    "Problem-solving abilities"
  ]
}}

Generate curriculum for ALL {semesters} semesters with unique subjects for each. Return ONLY the JSON."""

    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # OpenAI-compatible API format for Hugging Face Router
    payload = {
        "model": MODEL_ID,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": 3000,
        "temperature": 0.2,  # Lower for more consistent JSON
        "top_p": 0.9
    }
    
    try:
        print("📡 Sending request to Hugging Face Router API...")
        start_time = time.time()
        
        response = requests.post(
            HF_API_URL,
            headers=headers,
            json=payload,
            timeout=180
        )
        
        elapsed = time.time() - start_time
        print(f"⏱️  Response time: {elapsed:.2f} seconds")
        print(f"📊 HTTP Status: {response.status_code}")
        
        # Check for HTTP errors
        if response.status_code != 200:
            error_text = response.text
            print(f"❌ API Error ({response.status_code}): {error_text[:200]}")
            return {
                "error": f"API Error {response.status_code}",
                "details": error_text[:300],
                "status": "failed"
            }
        
        result = response.json()
        
        # Handle OpenAI-compatible response format
        if "choices" not in result:
            print(f"❌ Unexpected response format (no 'choices' field)")
            print(f"📦 Response keys: {list(result.keys())}")
            return {
                "error": "Invalid API response format",
                "details": f"Expected 'choices' field, got {list(result.keys())}",
                "status": "failed"
            }
        
        if len(result["choices"]) == 0:
            print(f"❌ No choices in response")
            return {
                "error": "Empty response from model",
                "details": "Model returned no output",
                "status": "failed"
            }
        
        # Extract generated text
        message = result["choices"][0].get("message", {})
        generated_text = message.get("content", "")
        
        if not generated_text:
            print(f"❌ No content in message")
            return {
                "error": "No content generated",
                "details": "Model returned empty content",
                "status": "failed"
            }
        
        print(f"✅ Response received!")
        print(f"📝 Generated text length: {len(generated_text)} characters")
        
        # Extract JSON from response
        json_start = generated_text.find('{')
        json_end = generated_text.rfind('}') + 1
        
        if json_start == -1 or json_end <= json_start:
            print(f"❌ No JSON found in response")
            print(f"📄 First 300 chars: {generated_text[:300]}")
            return {
                "error": "No valid JSON in response",
                "details": "Model did not return JSON format",
                "status": "failed",
                "response_preview": generated_text[:200]
            }
        
        json_str = generated_text[json_start:json_end]
        print(f"🔍 Extracted JSON: {len(json_str)} characters")
        
        # Parse JSON
        try:
            curriculum = json.loads(json_str)
            print(f"✅ JSON parsed successfully!")
            num_semesters = len(curriculum.get('semesters', []))
            print(f"✅ Semesters generated: {num_semesters}")
            print(f"✅ Skill: {curriculum.get('skill')}")
            print(f"✅ Level: {curriculum.get('level')}")
            print(f"{'='*70}")
            print(f"✨ CURRICULUM GENERATION SUCCESSFUL!")
            print(f"{'='*70}\n")
            return curriculum
            
        except json.JSONDecodeError as je:
            print(f"❌ JSON Parse Error: {str(je)}")
            print(f"📍 Error at line {je.lineno}, column {je.colno}")
            
            # Try fixing common issues
            json_str_fixed = json_str.replace("'", '"').replace('\n', ' ')
            try:
                curriculum = json.loads(json_str_fixed)
                print(f"✅ JSON fixed after cleanup!")
                return curriculum
            except:
                return {
                    "error": "Invalid JSON format",
                    "details": f"JSON parse error at position {je.pos}",
                    "status": "failed"
                }
    
    except requests.exceptions.Timeout:
        print(f"❌ Request timed out (>180 seconds)")
        return {
            "error": "Request Timeout",
            "details": "API took too long to respond. Try fewer semesters.",
            "status": "timeout"
        }
    except requests.exceptions.ConnectionError as e:
        print(f"❌ Connection Error: {str(e)}")
        return {
            "error": "Connection Error",
            "details": "Cannot reach Hugging Face API. Check internet connection.",
            "status": "connection_error"
        }
    except requests.exceptions.RequestException as e:
        print(f"❌ Request Exception: {str(e)}")
        return {
            "error": "Request Failed",
            "details": str(e)[:200],
            "status": "request_failed"
        }
    except Exception as e:
        print(f"❌ Unexpected Error: {type(e).__name__}: {str(e)}")
        return {
            "error": "Server Error",
            "details": str(e)[:200],
            "status": "server_error"
        }

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')

@app.route('/api/generate-curriculum', methods=['POST'])
def api_generate_curriculum():
    """API endpoint to generate curriculum"""
    try:
        data = request.json
        
        # Extract and validate inputs
        skill = data.get('skill', '').strip()
        level = data.get('level', '').strip()
        semesters = data.get('semesters', '')
        weekly_hours = data.get('weekly_hours', '').strip()
        industry_focus = data.get('industry_focus', '').strip()
        
        # Validation
        if not skill:
            return jsonify({"error": "Skill name is required"}), 400
        
        if not level:
            return jsonify({"error": "Education level is required"}), 400
        
        try:
            semesters = int(semesters)
            if semesters < 2 or semesters > 12:
                return jsonify({"error": "Semesters must be between 2 and 12"}), 400
        except (ValueError, TypeError):
            return jsonify({"error": "Invalid semester count"}), 400
        
        # Generate curriculum
        curriculum = generate_curriculum(skill, level, semesters, weekly_hours, industry_focus)
        
        # Check for errors
        if "error" in curriculum:
            return jsonify(curriculum), 500
        
        return jsonify(curriculum), 200
        
    except Exception as e:
        print(f"❌ API Handler Error: {str(e)}")
        return jsonify({
            "error": "Server Error",
            "details": str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model": MODEL_ID,
        "api": "Hugging Face Router",
        "timestamp": time.time()
    }), 200

if __name__ == '__main__':
    print(f"\n{'='*70}")
    print(f"🌐 Starting Flask Server")
    print(f"{'='*70}")
    print(f"🔗 URL: http://localhost:5000")
    print(f"🔗 Health: http://localhost:5000/health")
    print(f"📝 API: POST http://localhost:5000/api/generate-curriculum")
    print(f"{'='*70}\n")
    app.run(debug=True, host='0.0.0.0', port=5000)