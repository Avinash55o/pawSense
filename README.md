# Dog Breed Classifier

A web application that uses deep learning to identify dog breeds from uploaded images. Built with Next.js, FastAPI, and PyTorch.

## Features

- Upload and analyze dog photos
- Real-time breed classification
- Detailed breed information and characteristics
- Modern, responsive UI
- Dark mode support

## Tech Stack

### Frontend
- Next.js 13
- TypeScript
- Tailwind CSS
- Radix UI Components
- React Hook Form

### Backend
- FastAPI
- PyTorch
- TorchVision
- Python 3.8+

## Getting Started

### Prerequisites
- Node.js 16+
- Python 3.8+
- pip
- npm or yarn

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/dog-breed-classifier.git
cd dog-breed-classifier
```

2. Install frontend dependencies:
```bash
cd frontend
npm install
# or
yarn install
```

3. Install backend dependencies:
```bash
cd backend
pip install -r requirements.txt
```

### Running the Application

1. Start the backend server:
```bash
cd backend
uvicorn main:app --reload
```

2. Start the frontend development server:
```bash
cd frontend
npm run dev
# or
yarn dev
```

3. Open [http://localhost:3000](http://localhost:3000) in your browser.

## Usage

1. Click the "Upload Dog Image" button to select a photo
2. Wait for the analysis to complete
3. View the predicted breed(s) and their confidence scores
4. Read detailed information about the identified breed(s)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 