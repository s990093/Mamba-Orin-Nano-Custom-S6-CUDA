# Mamba2 Chat Frontend

Modern chat interface for Mamba2 MLX with comprehensive metrics visualization.

## Features

- 💬 **Real-time Chat Interface** - Clean, responsive chat UI
- 📊 **Speed Metrics** - Visualize prefill/decode speed and latency
- 💾 **Memory Tracking** - Monitor GPU memory usage
- 📈 **Quality Metrics** - View perplexity, log probabilities, and repetitions
- 🎯 **Top-K Candidates** - Step-by-step visualization of token probabilities
- ⚙️ **Configurable Parameters** - Adjust temperature, top-k, top-p, repetition penalty

## Quick Start

```bash
# Install dependencies
yarn

# Run development server
yarn dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

## Environment Variables

Create a `.env.local` file:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Usage

1. **Start the FastAPI backend** (in project root):
   ```bash
   python api_server.py
   ```

2. **Start the frontend** (in `frontend/` directory):
   ```bash
   yarn dev
   ```

3. **Configure generation parameters**:
   - Click the settings icon (⚙️) to adjust:
     - Temperature (0.0-2.0)
     - Top-K sampling
     - Top-P/nucleus sampling
     - Repetition penalty
     - Max tokens
     - Enable/disable Top-K candidates visualization

4. **Send a message** and view comprehensive metrics by clicking "Show Metrics"

## Tech Stack

- **Next.js 14+** - React framework
- **TypeScript** - Type safety
- **Tailwind CSS** - Styling
- **Recharts** - Data visualization
- **Lucide React** - Icons

## Project Structure

```
frontend/
├── app/
│   ├── page.tsx              # Main chat page
│   ├── layout.tsx            # Root layout
│   └── globals.css          # Global styles
├── components/
│   ├── ChatInterface.tsx     # Main chat UI
│   ├── MessageItem.tsx       # Message display
│   ├── MetricsPanel.tsx      # Metrics tabs
│   ├── SpeedChart.tsx        # Speed visualization
│   ├── MemoryChart.tsx       # Memory visualization
│   ├── QualityMetrics.tsx    # Quality indicators
│   └── TopKCandidates.tsx    # Top-k probabilities
├── types/
│   └── api.ts                # TypeScript types
└── lib/
    └── api.ts                # API client
```

## Metrics Visualization

### Speed Metrics
- Prefill time and speed
- Decode time and speed
- Average latency per token
- Total generation time

### Memory Usage
- Device type (GPU/CPU)
- Initial, current, and peak memory
- Memory used during generation

### Quality Metrics
- Perplexity (lower is better)
- Average log probability
- Number of repeated tokens
- Most repeated tokens with counts

### Generation Steps (Top-K Candidates)
- Step-by-step token selection
- Probability distribution visualization
- Navigate through generation sequence
- Highlight selected tokens

## Build for Production

```bash
yarn build
yarn start
```

## Development

```bash
# Type checking
yarn type-check

# Linting
yarn lint
```
