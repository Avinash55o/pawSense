#!/usr/bin/env node

/**
 * Script to help clean up unused UI components
 * Run with: node cleanup.js
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

// Define the components that are actually used in the project
const USED_COMPONENTS = [
  'button',
  'card',
  'tabs',
  'input',
  'textarea',
  'theme-provider',
  // Keep these utility components that might be indirectly used
  'toast',
  'toaster',
];

// Get all component files
function getAllComponents() {
  const componentsDir = path.join(__dirname, 'components', 'ui');
  const themeProvider = path.join(__dirname, 'components', 'theme-provider.tsx');
  
  let allComponentFiles = [];
  
  try {
    // Get UI components
    if (fs.existsSync(componentsDir)) {
      allComponentFiles = fs.readdirSync(componentsDir)
        .filter(file => file.endsWith('.tsx'))
        .map(file => path.join(componentsDir, file));
    }
    
    // Add theme provider
    if (fs.existsSync(themeProvider)) {
      allComponentFiles.push(themeProvider);
    }
  } catch (error) {
    console.error('Error reading component files:', error);
  }
  
  return allComponentFiles;
}

// Check if a component is used
function isComponentUsed(filePath) {
  const fileName = path.basename(filePath, '.tsx');
  return USED_COMPONENTS.includes(fileName);
}

// Main function
function analyzeComponents() {
  const allComponents = getAllComponents();
  
  const usedComponents = [];
  const unusedComponents = [];
  
  allComponents.forEach(componentPath => {
    if (isComponentUsed(componentPath)) {
      usedComponents.push(componentPath);
    } else {
      unusedComponents.push(componentPath);
    }
  });
  
  console.log(`\n=== PawSense Component Analysis ===\n`);
  console.log(`Found ${allComponents.length} total components`);
  console.log(`${usedComponents.length} components are used`);
  console.log(`${unusedComponents.length} components can be removed\n`);
  
  console.log('=== Components in use ===');
  usedComponents.forEach(comp => {
    console.log(`✅ ${path.basename(comp)}`);
  });
  
  console.log('\n=== Components that can be removed ===');
  unusedComponents.forEach(comp => {
    console.log(`❌ ${path.basename(comp)}`);
  });
  
  console.log('\n=== To remove unused components, run: ===');
  console.log('node cleanup.js --remove');
}

// Remove unused components
function removeUnusedComponents() {
  const allComponents = getAllComponents();
  
  let removedCount = 0;
  
  allComponents.forEach(componentPath => {
    if (!isComponentUsed(componentPath)) {
      try {
        fs.unlinkSync(componentPath);
        console.log(`Removed: ${componentPath}`);
        removedCount++;
      } catch (error) {
        console.error(`Error removing ${componentPath}:`, error);
      }
    }
  });
  
  console.log(`\nRemoved ${removedCount} unused components.`);
}

// Handle command line arguments
if (process.argv.includes('--remove')) {
  console.log('Removing unused components...');
  removeUnusedComponents();
} else {
  analyzeComponents();
} 