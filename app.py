from flask import Flask, render_template, request, jsonify
import os

app = Flask(__name__)

# Serve the HTML page
@app.route('/')
def index():
    return render_template('index.html')

# Analyze audio
@app.route('/analyze', methods=['POST'])
def analyze():
    audio_file = request.files['audio']
    audio_file.save("temp_audio.wav")
    
    # Mock analysis (replace with real code)
    return jsonify({
        "risk_status": "Low Risk",
        "wetness_level": "Medium",
        "dry_cough_count": 2,
        "wet_cough_count": 1,
        "lung_health_index": 85,
    })

if __name__ == "__main__":
    app.run(debug=True)