/** @type {import('next').NextConfig} */
const nextConfig = {
  // Remove export output for development mode
  // output: 'export',
  eslint: {
    ignoreDuringBuilds: true,
  },
  images: { unoptimized: true },
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'http://localhost:8000/api/:path*' // Proxy to Backend
      }
    ]
  }
};

module.exports = nextConfig;
