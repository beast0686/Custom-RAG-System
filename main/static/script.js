const form = document.getElementById('query-form');
const input = document.getElementById('query-input');
const loader = document.getElementById('loader');
const docsContainer = document.getElementById('docs-container');
const graphSection = document.getElementById('graph-section');
const graphContainer = document.getElementById('graph-container');
const errorContainer = document.getElementById('error-container');
const exportBtn = document.getElementById('export-graph');
let graphSessionId = null;
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
  const width = graphContainer.clientWidth;
  const height = 600;
  d3.select('#graph-container').selectAll('*').remove();

  const svg = d3.select('#graph-container')
    .attr('width', width)
    .attr('height', height)
    .style('background', '#f9fafb');

  const color = d3.scaleOrdinal()
    .domain(['document', 'Person', 'Organization', 'Location', 'Concept', 'Technology'])
    .range(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']);

  // Map edges to use source and target for D3.js compatibility
  const links = edges.map(e => ({ source: e.from, target: e.to, relation: e.relation }));

  const simulation = d3.forceSimulation(nodes)
    .force('link', d3.forceLink(links).id(d => d.id).distance(100))
    .force('charge', d3.forceManyBody().strength(-300))
    .force('center', d3.forceCenter(width / 2, height / 2));

  const link = svg.append('g')
    .attr('class', 'links')
    .selectAll('line')
    .data(links)
    .enter()
    .append('line')
    .attr('class', 'link')
    .attr('stroke', '#9575cd')
    .attr('stroke-width', 2);

  const linkText = svg.append('g')
    .attr('class', 'link-labels')
    .selectAll('text')
    .data(links.filter(e => e.relation))
    .enter()
    .append('text')
    .attr('class', 'link-text')
    .attr('dy', -2)
    .text(d => d.relation)
    .attr('font-size', '11px')
    .attr('fill', '#4a4a4a')
    .style('font-family', 'Inter, sans-serif');

  const node = svg.append('g')
    .attr('class', 'nodes')
    .selectAll('g')
    .data(nodes)
    .enter()
    .append('g')
    .attr('class', 'node')
    .on('click', (event, d) => {
      // Highlight node and animate
      d3.select(event.currentTarget).select('circle')
        .transition()
        .duration(200)
        .attr('r', 12)
        .transition()
        .duration(200)
        .attr('r', 8);
    })
    .call(d3.drag()
      .on('start', dragstarted)
      .on('drag', dragged)
      .on('end', dragended));

  node.append('circle')
    .attr('r', 8)
    .attr('fill', d => color(d.group))
    .attr('stroke', d => d3.color(color(d.group)).darker(0.5))
    .attr('stroke-width', 3);

  node.append('text')
    .attr('dx', 12)
    .attr('dy', '.35em')
    .text(d => d.label)
    .style('font-size', '13px')
    .style('font-family', 'Inter, sans-serif')
    .style('fill', '#1a1a1a');

  node.append('title')
    .text(d => `${d.label} (${d.group})`);

  simulation.on('tick', () => {
    link
      .attr('x1', d => d.source.x)
      .attr('y1', d => d.source.y)
      .attr('x2', d => d.target.x)
      .attr('y2', d => d.target.y);

    node
      .attr('transform', d => `translate(${d.x},${d.y})`);

    linkText
      .attr('x', d => (d.source.x + d.target.x) / 2)
      .attr('y', d => (d.source.y + d.target.y) / 2);
  });

  // Auto-fit graph
  setTimeout(() => {
    const bounds = svg.node().getBBox();
    if (bounds.width > 0 && bounds.height > 0) {
      const scale = Math.min(width / bounds.width, height / bounds.height) * 0.8;
      svg.transition()
        .duration(400)
        .attr('transform', `scale(${scale}) translate(${-bounds.x + width/(2*scale)},${-bounds.y + height/(2*scale)})`);
    }
  }, 2000);

  function dragstarted(event, d) {
    if (!event.active) simulation.alphaTarget(0.3).restart();
    d.fx = d.x;
    d.fy = d.y;
  }

  function dragged(event, d) {
    d.fx = event.x;
    d.fy = event.y;
  }

  function dragended(event, d) {
    if (!event.active) simulation.alphaTarget(0);
    d.fx = null;
    d.fy = null;
  }

  // Show export button with animation
  exportBtn.style.display = 'inline-block';
  exportBtn.style.opacity = '0';
  exportBtn.style.transform = 'translateY(10px)';
  setTimeout(() => {
    exportBtn.style.opacity = '1';
    exportBtn.style.transform = 'translateY(0)';
  }, 500);

  exportBtn.onclick = () => {
    // Convert SVG to PNG
    const svgElement = graphContainer.querySelector('svg');
    const serializer = new XMLSerializer();
    const svgString = serializer.serializeToString(svgElement);
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    const img = new Image();
    const svgBlob = new Blob([svgString], { type: 'image/svg+xml;charset=utf-8' });
    const url = URL.createObjectURL(svgBlob);

    img.onload = () => {
      canvas.width = width;
      canvas.height = height;
      ctx.drawImage(img, 0, 0);
      const link = document.createElement('a');
      link.download = 'knowledge-graph.png';
      link.href = canvas.toDataURL('image/png');
      link.click();
      URL.revokeObjectURL(url);

      // Visual feedback
      exportBtn.style.transform = 'scale(0.95)';
      setTimeout(() => {
        exportBtn.style.transform = 'scale(1)';
      }, 150);
    };
    img.src = url;
  };
}

// Enhanced document card interaction
docsContainer.addEventListener('click', (e) => {
  const card = e.target.closest('.doc-card');
  if (card) {
    const nodeId = card.id.replace("doc-", "doc_");
    // Highlight corresponding node in D3.js graph
    d3.selectAll('.node')
      .filter(d => d.id === nodeId)
      .select('circle')
      .transition()
      .duration(200)
      .attr('r', 12)
      .transition()
      .duration(200)
      .attr('r', 8);

    // Visual feedback for card
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
