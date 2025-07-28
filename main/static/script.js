const form = document.getElementById('query-form');
const input = document.getElementById('query-input');
const loader = document.getElementById('loader');
const docsContainer = document.getElementById('docs-container');
const graphSection = document.getElementById('graph-section');
const graphContainer = document.getElementById('graph-container');
const errorContainer = document.getElementById('error-container');
const exportBtn = document.getElementById('export-graph');
let graphSessionId = null;
let currentNetwork = null;
const kSlider = document.getElementById('k-slider');
const kLabel = document.getElementById('k-value-label');

// Enhanced slider interaction
kSlider.addEventListener('input', () => {
  kLabel.innerText = kSlider.value;
  kLabel.style.transform = 'scale(1.2)';
  kLabel.style.color = 'var(--primary-deep)';
  setTimeout(() => {
    kLabel.style.transform = 'scale(1)';
    kLabel.style.color = 'var(--primary-deep)';
  }, 200);
});

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
  if (graphSessionId) navigator.sendBeacon(`/cleanup/${graphSessionId}`, '');
});

// Enhanced form submission with animations
form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const queryText = input.value.trim();
  if (!queryText) {
    input.style.borderColor = '#d32f2f';
    input.style.boxShadow = '0 0 20px rgba(244, 67, 54, 0.3)';
    setTimeout(() => {
      input.style.borderColor = 'transparent';
      input.style.boxShadow = '0 2px 8px rgba(0,0,0,0.03)';
    }, 2000);
    return;
  }

  clearError();

  // Cleanup previous session
  if (graphSessionId) {
    try {
      await fetch(`/cleanup/${graphSessionId}`, { method: 'POST' });
    } catch (cleanupErr) {
      console.warn('Cleanup failed:', cleanupErr);
    }
    graphSessionId = null;
  }

  // Show loader with animation
  loader.style.display = 'block';
  loader.style.animation = 'fadeInScale 0.5s cubic-bezier(0.4, 0, 0.2, 1)';

  // Hide previous results with fade out
  const answerContainer = document.getElementById("answer-container");
  const answerText = document.getElementById("answer-text");

  if (answerContainer.style.display !== 'none') {
    answerContainer.style.opacity = '0';
    answerContainer.style.transform = 'translateY(-20px)';
    setTimeout(() => {
      answerContainer.style.display = 'none';
      answerText.innerHTML = '';
    }, 300);
  }

  if (docsContainer.innerHTML) {
    docsContainer.style.opacity = '0';
    docsContainer.style.transform = 'translateY(-20px)';
    setTimeout(() => {
      docsContainer.innerHTML = '';
    }, 300);
  }

  if (graphSection.style.display !== 'none') {
    graphSection.style.opacity = '0';
    graphSection.style.transform = 'translateY(-20px)';
    setTimeout(() => {
      graphSection.style.display = 'none';
      graphContainer.innerHTML = '';
      exportBtn.style.display = 'none';
    }, 300);
  }

  const k = parseInt(kSlider.value || 10);

  try {
    const response = await fetch('/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: queryText, k: k })
    });

    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.error || `HTTP error! Status: ${response.status}`);
    }

    const data = await response.json();

    // Show answer with animation
    if (data.answer) {
      answerText.innerText = data.answer.trim();
      answerContainer.style.display = 'block';
      answerContainer.style.opacity = '0';
      answerContainer.style.transform = 'translateY(20px)';
      setTimeout(() => {
        answerContainer.style.opacity = '1';
        answerContainer.style.transform = 'translateY(0)';
      }, 100);
    }

    graphSessionId = data.session_id;

    // Show documents with staggered animation
    if (data.retrieved_docs?.length > 0) {
      docsContainer.innerHTML = `<h2>Retrieved Documents</h2>`;
      docsContainer.style.opacity = '1';
      docsContainer.style.transform = 'translateY(0)';

      data.retrieved_docs.forEach((doc, idx) => {
        const card = document.createElement('div');
        card.className = 'doc-card';
        card.id = `doc-${doc.id}`;
        card.style.opacity = '0';
        card.style.transform = 'translateY(30px)';
        card.innerHTML = `
          <div class="doc-card-header">Document ${idx + 1} | Similarity Score: ${doc.score}</div>
          <div class="doc-card-content">${doc.summary}</div>
          <div class="doc-card-reference">Reference: <a href="${doc.url}" target="_blank" rel="noopener noreferrer">View Source</a></div>
        `;
        docsContainer.appendChild(card);

        // Staggered animation
        setTimeout(() => {
          card.style.opacity = '1';
          card.style.transform = 'translateY(0)';
        }, 100 + idx * 100);
      });
    } else if (data.answer) {
      // Answer only, no documents
    } else {
      docsContainer.innerHTML = '<h2>No relevant documents found for this query.</h2>';
      docsContainer.style.opacity = '1';
      docsContainer.style.transform = 'translateY(0)';
    }

    // Show graph with animation
    if (data.nodes?.length > 0) {
      graphSection.style.display = 'block';
      graphSection.style.opacity = '0';
      graphSection.style.transform = 'translateY(20px)';
      setTimeout(() => {
        graphSection.style.opacity = '1';
        graphSection.style.transform = 'translateY(0)';
        renderGraph(data.nodes, data.edges);
      }, 200);
    } else if (data.retrieved_docs?.length > 0) {
      graphSection.style.display = 'block';
      graphSection.style.opacity = '0';
      graphSection.style.transform = 'translateY(20px)';
      setTimeout(() => {
        graphSection.style.opacity = '1';
        graphSection.style.transform = 'translateY(0)';
        graphContainer.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:var(--text-secondary);font-size:1.125rem;">Documents were found, but no entities could be extracted to build a graph.</div>';
      }, 200);
    }

  } catch (error) {
    showError(error.message || "Unknown error occurred");
  } finally {
    // Hide loader with animation
    loader.style.opacity = '0';
    setTimeout(() => {
      loader.style.display = 'none';
      loader.style.opacity = '1';
    }, 300);
  }
});

function showError(msg) {
  errorContainer.textContent = msg;
  errorContainer.style.display = 'block';
  errorContainer.style.animation = 'shake 0.5s ease-in-out';
}

function clearError() {
  errorContainer.textContent = '';
  errorContainer.style.display = 'none';
}

function renderGraph(nodes, edges) {
  const graphData = {
    nodes: new vis.DataSet(nodes.map(n => ({
      ...n,
      shape: 'dot',
      size: 25,
      title: `${n.label} (${n.group})`,
      font: { size: 13, color: '#1a1a1a', strokeWidth: 2, strokeColor: '#ffffff' }
    }))),
    edges: new vis.DataSet(edges.map(e => ({
      ...e,
      arrows: 'to',
      label: e.relation || '',
      font: { align: 'top', size: 11, color: '#4a4a4a', strokeWidth: 1, strokeColor: '#ffffff' }
    })))
  };

  const options = {
    nodes: {
      font: { size: 13, color: '#1a1a1a' },
      borderWidth: 3,
      shadow: { enabled: true, color: 'rgba(0,0,0,0.15)', size: 8, x: 2, y: 2 }
    },
    edges: {
      width: 2,
      color: { color: '#9575cd', highlight: '#6a1b9a' },
      smooth: { type: 'dynamic' },
      shadow: { enabled: true, color: 'rgba(0,0,0,0.1)', size: 4, x: 1, y: 1 }
    },
    physics: {
      forceAtlas2Based: {
        gravitationalConstant: -30,
        centralGravity: 0.01,
        springLength: 200,
        springConstant: 0.04
      },
      solver: 'forceAtlas2Based',
      timestep: 0.4,
      stabilization: { iterations: 150 }
    },
    groups: {
      document: { color: { background: '#FF6B6B', border: '#FF5252' }, shape: 'dot' },
      Person: { color: { background: '#4ECDC4', border: '#26A69A' }, shape: 'dot' },
      Organization: { color: { background: '#45B7D1', border: '#2196F3' }, shape: 'dot' },
      Location: { color: { background: '#96CEB4', border: '#66BB6A' }, shape: 'dot' },
      Concept: { color: { background: '#FFEAA7', border: '#FFEB3B' }, shape: 'dot' },
      Technology: { color: { background: '#DDA0DD', border: '#BA68C8' }, shape: 'dot' }
    }
  };

  currentNetwork = new vis.Network(graphContainer, graphData, options);

  // Show export button with animation
  exportBtn.style.display = 'inline-block';
  exportBtn.style.opacity = '0';
  exportBtn.style.transform = 'translateY(10px)';
  setTimeout(() => {
    exportBtn.style.opacity = '1';
    exportBtn.style.transform = 'translateY(0)';
  }, 500);

  exportBtn.onclick = () => {
    const canvas = graphContainer.getElementsByTagName("canvas")[0];
    const link = document.createElement("a");
    link.download = "knowledge-graph.png";
    link.href = canvas.toDataURL();
    link.click();

    // Visual feedback
    exportBtn.style.transform = 'scale(0.95)';
    setTimeout(() => {
      exportBtn.style.transform = 'scale(1)';
    }, 150);
  };
}

// Enhanced document card interaction
docsContainer.addEventListener('click', (e) => {
  const card = e.target.closest('.doc-card');
  if (card && currentNetwork) {
    const nodeId = card.id.replace("doc-", "doc_");
    currentNetwork.selectNodes([nodeId]);
    currentNetwork.focus(nodeId, { scale: 1.5, animation: { duration: 1000, easingFunction: 'easeInOutQuad' } });

    // Visual feedback
    card.style.transform = 'translateY(-8px) scale(1.02)';
    card.style.boxShadow = 'var(--shadow-strong), var(--shadow-glow)';
    setTimeout(() => {
      card.style.transform = 'translateY(-4px)';
      card.style.boxShadow = 'var(--shadow-strong)';
    }, 200);
  }
});

// Enhanced input interactions
input.addEventListener('focus', () => {
  input.parentElement.style.transform = 'translateY(-2px)';
});

input.addEventListener('blur', () => {
  input.parentElement.style.transform = 'translateY(0)';
});

// Add subtle parallax effect on scroll
window.addEventListener('scroll', () => {
  const scrolled = window.pageYOffset;
  const rate = scrolled * -0.3;
  document.body.style.backgroundPosition = `0 ${rate}px`;
});
