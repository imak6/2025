// @ts-check

/**
 * Sidebars configuration for Docusaurus.
 *
 * This sidebar groups together related documentation pages
 * under "AI Social Companion".
 *
 * @type {import('@docusaurus/plugin-content-docs').SidebarsConfig}
 */

const sidebars = {
    tutorialSidebar: [
      {
        type: 'category',
        label: 'AI Social Companion',
        collapsed: false,
        items: [
          'ai-social-companion',     // maps to docs/ai-social-companion.md
          'planning',                // maps to docs/planning.md (if it exists)
        ],
      },
    ],
  };

  module.exports = sidebars;
