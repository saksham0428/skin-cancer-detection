import { useState, useRef, type ChangeEvent, type DragEvent } from 'react';
import axios, { type AxiosError } from 'axios';
import { UploadCloud, AlertCircle, Loader2, RefreshCw, Info } from 'lucide-react';
import clsx from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs: (string | undefined | null | false)[]) {
  return twMerge(clsx(inputs));
}

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

interface PredictionResponse {
  predicted_class: string;
  predicted_class_full_name: string;
  class_index: number;
  confidence: number;
  probabilities: Record<string, number>;
  disclaimer: string;
}

const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB
const ALLOWED_TYPES = ['image/jpeg', 'image/png', 'image/webp', 'image/bmp'];

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (selectedFile: File | null) => {
    setError(null);
    setResult(null);
    
    if (!selectedFile) return;

    if (!ALLOWED_TYPES.includes(selectedFile.type)) {
      setError('Unsupported file type. Please upload a JPEG, PNG, WebP, or BMP image.');
      return;
    }

    if (selectedFile.size > MAX_FILE_SIZE) {
      setError('File is too large. Maximum allowed size is 10MB.');
      return;
    }

    setFile(selectedFile);
    
    const reader = new FileReader();
    reader.onload = (e) => {
      setPreview(e.target?.result as string);
    };
    reader.readAsDataURL(selectedFile);
  };

  const onFileSelect = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFileChange(e.target.files[0]);
    }
  };

  const onDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const onDragLeave = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const onDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileChange(e.dataTransfer.files[0]);
    }
  };

  const handleAnalyze = async () => {
    if (!file) return;
    
    setIsLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post<PredictionResponse>(`${API_URL}/predict`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setResult(response.data);
    } catch (err) {
      const axiosError = err as AxiosError<{detail: string, error: string}>;
      if (!axiosError.response) {
        setError('Cannot connect to the server. Please make sure the backend is running.');
      } else {
        const msg = axiosError.response.data?.detail || 'An error occurred during analysis.';
        setError(msg);
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 font-sans selection:bg-blue-200">
      <header className="bg-white border-b border-slate-200 shadow-sm sticky top-0 z-10">
        <div className="max-w-6xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="bg-blue-600 text-white p-2 rounded-lg">
              <UploadCloud className="w-6 h-6" />
            </div>
            <h1 className="text-xl font-semibold tracking-tight text-slate-800">
              Skin Lesion Analysis Dashboard
            </h1>
          </div>
          <div className="text-sm text-slate-500 font-medium">Research & Education Edition</div>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-4 py-8">
        
        {/* Disclaimer */}
        <div className="mb-8 bg-amber-50 border border-amber-200 rounded-lg p-4 flex items-start gap-3 text-amber-800 shadow-sm">
          <AlertCircle className="w-5 h-5 shrink-0 mt-0.5" />
          <div className="text-sm">
            <strong className="font-semibold block mb-1">Medical Disclaimer</strong>
            This tool is for research and educational purposes only. It is not a medical diagnosis tool. 
            Do not use these results to make clinical decisions. Consult a qualified healthcare professional for medical advice.
          </div>
        </div>

        <div className={cn("grid gap-8 items-start", result ? "lg:grid-cols-2" : "lg:grid-cols-1 max-w-2xl mx-auto")}>
          
          {/* Left Column: Upload / Preview */}
          <div className="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden flex flex-col">
            <div className="p-5 border-b border-slate-100 bg-slate-50/50">
              <h2 className="text-lg font-medium text-slate-800">Image Input</h2>
            </div>
            
            <div className="p-6 grow flex flex-col">
              {!preview ? (
                <div 
                  className={cn(
                    "border-2 border-dashed rounded-xl p-10 flex flex-col items-center justify-center text-center transition-colors grow",
                    isDragging ? "border-blue-500 bg-blue-50" : "border-slate-300 hover:bg-slate-50"
                  )}
                  onDragOver={onDragOver}
                  onDragLeave={onDragLeave}
                  onDrop={onDrop}
                  onClick={() => fileInputRef.current?.click()}
                  style={{ cursor: 'pointer' }}
                >
                  <div className="bg-blue-100 text-blue-600 p-4 rounded-full mb-4">
                    <UploadCloud className="w-8 h-8" />
                  </div>
                  <h3 className="text-lg font-medium text-slate-700 mb-1">Drag and drop an image</h3>
                  <p className="text-slate-500 text-sm mb-6">or click to browse from your computer</p>
                  <p className="text-xs text-slate-400">Supports JPEG, PNG, WebP, BMP up to 10MB</p>
                  <input 
                    type="file" 
                    className="hidden" 
                    ref={fileInputRef} 
                    onChange={onFileSelect}
                    accept="image/jpeg, image/png, image/webp, image/bmp"
                    data-testid="file-upload"
                  />
                </div>
              ) : (
                <div className="flex flex-col items-center gap-4 grow">
                  <div className="relative w-full max-w-sm rounded-lg overflow-hidden border border-slate-200 shadow-sm bg-slate-100 aspect-square flex items-center justify-center">
                    <img src={preview} alt="Lesion preview" className="max-w-full max-h-full object-contain" />
                  </div>
                  
                  <div className="flex gap-3 w-full max-w-sm mt-4">
                    {!result && (
                      <button 
                        onClick={handleAnalyze}
                        disabled={isLoading}
                        className="flex-1 bg-blue-600 hover:bg-blue-700 disabled:bg-blue-400 text-white font-medium py-2.5 px-4 rounded-lg flex items-center justify-center gap-2 transition-colors shadow-sm"
                      >
                        {isLoading ? (
                          <><Loader2 className="w-5 h-5 animate-spin" /> Analyzing...</>
                        ) : (
                          'Analyze Image'
                        )}
                      </button>
                    )}
                    <button 
                      onClick={handleReset}
                      disabled={isLoading}
                      className={cn(
                        "flex items-center justify-center gap-2 font-medium py-2.5 px-4 rounded-lg transition-colors border shadow-sm",
                        result 
                          ? "flex-1 bg-white border-slate-300 text-slate-700 hover:bg-slate-50" 
                          : "px-4 bg-white border-slate-300 text-slate-700 hover:bg-slate-50"
                      )}
                    >
                      <RefreshCw className="w-4 h-4" />
                      {result ? 'Analyze Another Image' : 'Clear'}
                    </button>
                  </div>
                </div>
              )}

              {error && (
                <div className="mt-6 p-4 bg-red-50 text-red-700 rounded-lg border border-red-200 flex items-start gap-3 text-sm">
                  <AlertCircle className="w-5 h-5 shrink-0 mt-0.5" />
                  <span>{error}</span>
                </div>
              )}
            </div>
          </div>

          {/* Right Column: Results */}
          {result && (
            <div className="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden flex flex-col animate-in fade-in slide-in-from-bottom-4 duration-500 fill-mode-both">
              <div className="p-5 border-b border-slate-100 bg-slate-50/50 flex justify-between items-center">
                <h2 className="text-lg font-medium text-slate-800">Analysis Results</h2>
                <div className="flex items-center gap-1.5 text-xs font-medium px-2.5 py-1 bg-emerald-100 text-emerald-700 rounded-full">
                  <span className="relative flex h-2 w-2">
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                    <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
                  </span>
                  Analysis Complete
                </div>
              </div>
              
              <div className="p-6">
                <div className="mb-8 text-center p-6 bg-slate-50 rounded-xl border border-slate-100">
                  <p className="text-sm text-slate-500 uppercase tracking-wider font-semibold mb-1">Primary Prediction</p>
                  <h3 className="text-3xl font-bold text-slate-800 mb-2">{result.predicted_class_full_name}</h3>
                  <div className="flex items-center justify-center gap-2">
                    <span className="text-slate-500 bg-slate-200/50 px-2.5 py-0.5 rounded text-sm font-medium">Class: {result.predicted_class.toUpperCase()}</span>
                    <span className="text-blue-700 bg-blue-100 px-2.5 py-0.5 rounded text-sm font-medium">
                      Confidence: {(result.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>

                <div>
                  <h4 className="text-sm font-semibold text-slate-800 mb-4 uppercase tracking-wider flex items-center gap-2">
                    Probability Distribution
                  </h4>
                  <div className="space-y-4">
                    {Object.entries(result.probabilities)
                      .sort(([, a], [, b]) => b - a)
                      .map(([className, prob]) => (
                        <div key={className} className="relative">
                          <div className="flex justify-between text-sm mb-1">
                            <span className={cn("font-medium", className === result.predicted_class ? "text-blue-700" : "text-slate-600")}>
                              {className.toUpperCase()}
                            </span>
                            <span className="text-slate-500 font-mono">{(prob * 100).toFixed(1)}%</span>
                          </div>
                          <div className="h-2 w-full bg-slate-100 rounded-full overflow-hidden">
                            <div 
                              className={cn(
                                "h-full rounded-full transition-all duration-1000 ease-out",
                                className === result.predicted_class ? "bg-blue-500" : "bg-slate-300"
                              )}
                              style={{ width: `${Math.max(prob * 100, 1)}%` }}
                            />
                          </div>
                        </div>
                      ))}
                  </div>
                </div>
                
                <div className="mt-8 flex items-start gap-2 text-xs text-slate-500 bg-slate-50 p-3 rounded-lg">
                  <Info className="w-4 h-4 shrink-0 mt-0.5" />
                  <p>{result.disclaimer || "Generated by research AI model."}</p>
                </div>
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

export default App;
