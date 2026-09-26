import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach, type Mocked } from 'vitest';
import axios from 'axios';
import App from './App';

vi.mock('axios');
const mockedAxios = axios as Mocked<typeof axios>;

// Mock URL.createObjectURL for the test environment
globalThis.URL.createObjectURL = vi.fn(() => 'blob:mock-url');

describe('App Component', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders the dashboard and disclaimer', () => {
    render(<App />);
    expect(screen.getByText(/SKYNEX/i)).toBeInTheDocument();
    expect(screen.getByText(/See Beyond the Surface/i)).toBeInTheDocument();
  });

  it('shows error for unsupported file type', async () => {
    render(<App />);
    const file = new File(['dummy content'], 'test.txt', { type: 'text/plain' });
    const input = screen.getByTestId('file-upload');
    
    fireEvent.change(input, { target: { files: [file] } });
    
    expect(await screen.findByText(/Unsupported file type/i)).toBeInTheDocument();
  });

  it('shows error for oversized file', async () => {
    render(<App />);
    const file = new File(['x'.repeat(11 * 1024 * 1024)], 'large.jpg', { type: 'image/jpeg' });
    const input = screen.getByTestId('file-upload');
    
    fireEvent.change(input, { target: { files: [file] } });
    
    expect(await screen.findByText(/File is too large/i)).toBeInTheDocument();
  });

  it('performs end-to-end analysis and displays results', async () => {
    mockedAxios.post.mockResolvedValueOnce({
      data: {
        predicted_class: 'mel',
        predicted_class_full_name: 'Melanoma',
        class_index: 4,
        confidence: 0.9543,
        probabilities: {
          akiec: 0.01, bcc: 0.01, bkl: 0.01, df: 0.01, mel: 0.9543, nv: 0.0057, vasc: 0.00
        },
        disclaimer: 'Test disclaimer'
      }
    });

    render(<App />);
    
    // 1. Upload valid image
    const file = new File(['dummy'], 'lesion.jpg', { type: 'image/jpeg' });
    const input = screen.getByTestId('file-upload');
    fireEvent.change(input, { target: { files: [file] } });

    // 2. Click Analyze
    const analyzeBtn = await screen.findByRole('button', { name: /Analyze Image/i });
    fireEvent.click(analyzeBtn);

    // 3. Verify loading state
    expect(screen.getByText(/Analyzing\.\.\./i)).toBeInTheDocument();

    // 4. Wait for results
    await waitFor(() => {
      expect(screen.getByText(/Melanoma/i)).toBeInTheDocument();
    });

    // 5. Verify displayed data matches API response
    expect(screen.getByText(/Class: MEL/i)).toBeInTheDocument();
    expect(screen.getByText(/Confidence: 95.4%/i)).toBeInTheDocument();
    expect(screen.getByText(/Test disclaimer/i)).toBeInTheDocument();
  });

  it('handles API errors gracefully', async () => {
    mockedAxios.post.mockRejectedValueOnce({
      response: { data: { detail: 'Model failed to process image' } }
    });

    render(<App />);
    
    const file = new File(['dummy'], 'lesion.jpg', { type: 'image/jpeg' });
    const input = screen.getByTestId('file-upload');
    fireEvent.change(input, { target: { files: [file] } });

    const analyzeBtn = await screen.findByRole('button', { name: /Analyze Image/i });
    fireEvent.click(analyzeBtn);

    expect(await screen.findByText(/Model failed to process image/i)).toBeInTheDocument();
  });

  it('handles backend unavailable errors gracefully', async () => {
    mockedAxios.post.mockRejectedValueOnce(new Error('Network Error'));

    render(<App />);
    
    const file = new File(['dummy'], 'lesion.jpg', { type: 'image/jpeg' });
    const input = screen.getByTestId('file-upload');
    fireEvent.change(input, { target: { files: [file] } });

    const analyzeBtn = await screen.findByRole('button', { name: /Analyze Image/i });
    fireEvent.click(analyzeBtn);

    expect(await screen.findByText(/Cannot connect to the server/i)).toBeInTheDocument();
  });
});
