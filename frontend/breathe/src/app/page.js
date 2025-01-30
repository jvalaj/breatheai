'use client'
import { useState, useEffect } from 'react';

export default function Home() {
  const [isRecording, setIsRecording] = useState(false);
  const [results, setResults] = useState(null);
  const [mediaRecorder, setMediaRecorder] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    navigator.mediaDevices.getUserMedia({ audio: true })
      .then(stream => {
        const recorder = new MediaRecorder(stream);
        let audioChunks = [];

        recorder.ondataavailable = e => {
          audioChunks.push(e.data);
        };

        recorder.onstop = async () => {
          const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
          audioChunks = [];
          analyzeAudio(audioBlob);
        };

        setMediaRecorder(recorder);
      });
  }, []);

  const analyzeAudio = async (blob) => {
    setLoading(true);
    const formData = new FormData();
    formData.append('audio', blob, 'recording.wav');

    try {
      const response = await fetch('/api/analyze', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      setResults(data);
    } catch (error) {
      console.error('Analysis error:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-black text-gray-100 p-8">
      <div className="max-w-3xl mx-auto">
        <h1 className="text-4xl font-extralight mb-8 text-cyan-400">breathe.</h1>

        {/* Recording Section */}
        <button
          onClick={() => {
            isRecording ? mediaRecorder.stop() : mediaRecorder.start();
            setIsRecording(!isRecording);
          }}
          className={`p-3 rounded-md font-medium transition-all
              ${isRecording
              ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-400/30 hover:bg-cyan-500/30'
              : 'bg-cyan-500/10 text-cyan-300 border border-cyan-400/20 hover:bg-cyan-500/20'}
            `}
        >
          {isRecording ? '⏹ stop' : '🎤 start'}
        </button>

        {/* Results Section */}
        {loading && (
          <div className="text-center py-8">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-cyan-400 mx-auto"></div>
          </div>
        )}

        {results && (
          <div className="bg-gray-800/50 rounded-xl p-6 backdrop-blur-lg">
            <h2 className="text-2xl font-semibold mb-6 text-cyan-300">Analysis Results</h2>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <MetricCard title="Risk Status" value={results.risk_status} color="cyan" />
              <MetricCard title="Lung Health Index" value={results.lung_health_index} unit="%" />
              <MetricCard title="Dry Coughs" value={results.dry_cough_count} />
              <MetricCard title="Wet Coughs" value={results.wet_cough_count} />
              <MetricCard title="Wetness Level" value={results.wetness_level} />
              <MetricCard title="Pattern Severity" value={results.pattern_severity} />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

const MetricCard = ({ title, value, unit, color = 'gray' }) => (
  <div className={`p-4 rounded-lg bg-${color}-500/10 border border-${color}-400/20`}>
    <h3 className="text-sm text-gray-400 mb-1">{title}</h3>
    <p className="text-2xl font-medium">
      {value}
      {unit && <span className="text-sm ml-1 text-gray-400">{unit}</span>}
    </p>
  </div>
);