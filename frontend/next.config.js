/** @type {import('next').NextConfig} */
const nextConfig = {
  // Output in standalone mode for Docker
  output: 'standalone',
  eslint: {
    ignoreDuringBuilds: true,
  },
  images: { unoptimized: true },
  async rewrites() {
    const API_URL = process.env.BACKEND_URL || 
                   process.env.NEXT_PUBLIC_API_URL || 
                   'http://localhost:8000';
    
    console.log(`Using API URL: ${API_URL}`);
    
    return [
      {
        source: '/api/:path*',
        destination: `${API_URL}/api/:path*` // Proxy to Backend
      }
    ]
  }
};

module.exports = nextConfig;
