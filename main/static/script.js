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
const compareBtn = document.getElementById('compare-btn');
const comparisonSection = document.getElementById('comparison-section');
const comparisonLoader = document.getElementById('comparison-loader');
const comparisonResultsContainer = document.getElementById('comparison-results-container');
const plainLlmAnswer = document.getElementById('plain-llm-answer');
const mongodbRagAnswer = document.getElementById('mongodb-rag-answer');
const neo4jKgRagAnswer = document.getElementById('neo4j-kg-rag-answer');

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

  if (graphSessionId) {
    try { await fetch(`/cleanup/${graphSessionId}`, { method: 'POST' }); }
    catch (cleanupErr) { console.warn('Cleanup failed:', cleanupErr); }
    graphSessionId = null;
  }

  loader.style.display = 'block';
  loader.style.animation = 'fadeInScale 0.5s cubic-bezier(0.4, 0, 0.2, 1)';

  // Hide previous results
  const answerContainer = document.getElementById("answer-container");
  const answerText = document.getElementById("answer-text");

  if (comparisonSection.style.display !== 'none') {
    comparisonSection.style.opacity = '0';
    comparisonSection.style.transform = 'translateY(-20px)';
    setTimeout(() => {
        comparisonSection.style.display = 'none';
        comparisonResultsContainer.style.display = 'none';
        compareBtn.textContent = 'Compare RAG Approaches';
        compareBtn.disabled = false;
    }, 300);
  }
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
    setTimeout(() => { docsContainer.innerHTML = ''; }, 300);
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
    comparisonSection.style.display = 'flex';
    comparisonSection.style.opacity = '1';
    comparisonSection.style.transform = 'translateY(0)';
    comparisonResultsContainer.style.display = 'none';

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
          <div class="doc-card-reference">Reference: <a href="${doc.url}" target="_blank" rel="noopener noreferrer">View Source</a></div>`;
        docsContainer.appendChild(card);
        setTimeout(() => {
          card.style.opacity = '1';
          card.style.transform = 'translateY(0)';
        }, 100 + idx * 100);
      });
    } else {
      docsContainer.innerHTML = '<h2>No relevant documents found for this query.</h2>';
      docsContainer.style.opacity = '1';
      docsContainer.style.transform = 'translateY(0)';
    }

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
        graphContainer.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:var(--text-secondary);font-size:1.125rem;">Documents were found, but no entities could be extracted.</div>';
      }, 200);
    }
  } catch (error) {
    showError(error.message || "Unknown error occurred");
  } finally {
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
    const container = document.getElementById('graph-container');
    container.innerHTML = '';

    if (!nodes || nodes.length === 0) {
        container.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:var(--text-secondary);font-size:1.125rem;">No data available to build a graph.</div>';
        return;
    }
    
    const graphBgColor = '#211f24';
    const textColor = '#f1f1f1';
    const linkLabelColor = '#000000';
    const linkLabelStroke = '#ffffff';

    const width = container.clientWidth;
    const height = container.clientHeight;

    const dbNode = { id: "db_center", label: "DB", group: "Database", fx: width / 2, fy: height / 2 };
    const finalNodes = [dbNode, ...nodes];
    const docLinks = nodes.filter(n => n.group === 'Document').map(docNode => ({ source: dbNode.id, target: docNode.id }));
    const finalEdges = [...edges, ...docLinks];
    const links = finalEdges.map(e => ({ source: e.from || e.source, target: e.to || e.target, relation: e.relation }));

    const svg = d3.select(container).append('svg')
        .attr('viewBox', `0 0 ${width} ${height}`)
        .style('background-color', graphBgColor);

    const g = svg.append("g");
    const color = d3.scaleOrdinal()
        .domain(['Database', 'Document', 'Person', 'Organization', 'Location', 'Concept', 'Technology', 'Event', 'Product', 'Inferred'])
        .range(['#ff7f0e', '#4e79a7', '#f28e2c', '#59a14f', '#e15759', '#76b7b2', '#af7aa1', '#ff9da7', '#9c755f', '#bab0ab']);

    // --- FIX: Re-tuned physics for a more spread-out and stable layout ---
    const simulation = d3.forceSimulation(finalNodes)
        .force("link", d3.forceLink(links).id(d => d.id).distance(d => d.relation ? 150 : 80).strength(0.4))
        .force("charge", d3.forceManyBody().strength(-900))
        .force("center", d3.forceCenter(width / 2, height / 2).strength(0.1))
        .force("collision", d3.forceCollide().radius(35));

    const link = g.append('g').selectAll('line').data(links).enter().append('line')
        .attr('stroke', d => d.relation ? '#999' : '#555')
        .attr('stroke-opacity', 0.8)
        .attr('stroke-width', d => d.relation ? 1.5 : 1);

    const linkText = g.append('g').selectAll('text').data(links).enter().append('text')
        .text(d => d.relation || '')
        .attr('class', 'link-label-text')
        .attr('font-size', '10px').attr('text-anchor', 'middle').style('font-weight', 'normal')
        .attr('fill', linkLabelColor).style('paint-order', 'stroke').style('stroke', linkLabelStroke)
        .style('stroke-width', '2px').style('stroke-linecap', 'round');

    const node = g.append('g').selectAll('g').data(finalNodes).enter().append('g')
        .attr('class', 'node')
        .style('cursor', 'pointer')
        .call(drag(simulation));

    node.on('click', (event, d) => {
        if (d.group === 'Document') {
            const cardId = d.id.replace('doc_', 'doc-');
            const cardElement = document.getElementById(cardId);
            if (cardElement) {
                cardElement.scrollIntoView({ behavior: 'smooth', block: 'center' });
                cardElement.style.transition = 'all 0.3s ease-in-out';
                cardElement.style.transform = 'scale(1.05)';
                cardElement.style.boxShadow = '0 0 20px rgba(78, 121, 167, 0.7)';
                setTimeout(() => {
                    cardElement.style.transform = 'scale(1)';
                    cardElement.style.boxShadow = '';
                }, 1500);
            }
        }
    });
    
    node.on('dblclick', (event, d) => {
        if (d.id !== 'db_center') {
            d.fx = null;
            d.fy = null;
            simulation.alpha(0.3).restart();
        }
    });

    node.each(function(d) {
        const group = d3.select(this);
        if (d.group === 'Database') group.append('circle').attr('r', 20).attr('fill', color(d.group)).attr('stroke', '#ffcc00').attr('stroke-width', 3);
        else if (d.group === 'Document') group.append('rect').attr('width', 18).attr('height', 18).attr('x', -9).attr('y', -9).attr('rx', 3).attr('fill', color(d.group)).attr('stroke', '#fff').attr('stroke-width', 2);
        else group.append('circle').attr('r', 10).attr('fill', color(d.group)).attr('stroke', '#fff').attr('stroke-width', 2);
    });

    node.append('text').text(d => d.label).attr('dy', '0.35em').attr('dx', d => d.group === 'Database' ? 25 : 14)
        .style('font-size', '12px').style('font-family', 'Inter, sans-serif').style('fill', textColor)
        .style('text-shadow', `0 0 5px ${graphBgColor}, 0 0 5px ${graphBgColor}`);

    node.append('title').text(d => `${d.label} (${d.group})`);

    simulation.on('tick', () => {
        link.attr('x1', d => d.source.x).attr('y1', d => d.source.y).attr('x2', d => d.target.x).attr('y2', d => d.target.y);
        node.attr('transform', d => `translate(${d.x},${d.y})`);
        linkText.attr('x', d => (d.source.x + d.target.x) / 2).attr('y', d => (d.source.y + d.target.y) / 2);
    });

    svg.call(d3.zoom().scaleExtent([0.2, 7]).on('zoom', (event) => g.attr('transform', event.transform)));

    function drag(simulation) {
        function dragstarted(event, d) { if (!event.active) simulation.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; }
        function dragged(event, d) { d.fx = event.x; d.fy = event.y; }
        function dragended(event, d) { if (!event.active) simulation.alphaTarget(0); }
        return d3.drag().on('start', dragstarted).on('drag', dragged).on('end', dragended);
    }

    exportBtn.style.display = 'inline-block';
    exportBtn.style.opacity = '0';
    exportBtn.style.transform = 'translateY(10px)';
    setTimeout(() => {
        exportBtn.style.opacity = '1';
        exportBtn.style.transform = 'translateY(0)';
    }, 500);

    exportBtn.onclick = () => {
        const currentTransform = d3.zoomTransform(svg.node());
        g.attr('transform', null);
        const bounds = g.node().getBBox();
        const padding = 40;
        const framedSvgString = `<svg width="${bounds.width + padding * 2}" height="${bounds.height + padding * 2}" viewBox="${bounds.x - padding} ${bounds.y - padding} ${bounds.width + padding * 2} ${bounds.height + padding * 2}" xmlns="http://www.w3.org/2000/svg">
            <style>
              .node text { font-family: Inter, sans-serif; font-size:12px; fill: ${textColor}; text-shadow: 0 0 5px ${graphBgColor}, 0 0 5px ${graphBgColor}; }
              .link-label-text { font-family: Inter, sans-serif; font-size: 10px; fill: ${linkLabelColor}; text-anchor: middle; paint-order: stroke; stroke: ${linkLabelStroke}; stroke-width: 2px; stroke-linecap: round; }
            </style>
            ${g.node().innerHTML}
          </svg>`;
        g.attr('transform', currentTransform);
        const svgBlob = new Blob([framedSvgString], { type: "image/svg+xml;charset=utf-8" });
        const url = URL.createObjectURL(svgBlob);
        const img = new Image();
        img.onload = () => {
            const scaleFactor = 3;
            const canvas = document.createElement('canvas');
            canvas.width = (bounds.width + padding * 2) * scaleFactor;
            canvas.height = (bounds.height + padding * 2) * scaleFactor;
            const ctx = canvas.getContext('2d');
            ctx.fillStyle = graphBgColor;
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
            const link = document.createElement('a');
            link.download = 'knowledge-graph.png';
            link.href = canvas.toDataURL('image/png');
            link.click();
            URL.revokeObjectURL(url);
        };
        img.src = url;
    };
}


compareBtn.addEventListener('click', async () => {
    if (!graphSessionId) {
        showError("Please perform a query first before comparing results.");
        return;
    }
    comparisonLoader.style.display = 'block';
    comparisonResultsContainer.style.display = 'none';
    compareBtn.disabled = true;
    compareBtn.textContent = 'Comparing...';
    try {
        const response = await fetch('/generate_comparison', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: graphSessionId })
        });
        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.error || `HTTP error! Status: ${response.status}`);
        }
        const data = await response.json();
        
        plainLlmAnswer.innerText = data.plain_llm_answer;
        mongodbRagAnswer.innerText = data.mongodb_rag_answer;
        neo4jKgRagAnswer.innerText = data.neo4j_kg_rag_answer;

        if (data.calculated_metrics) {
            document.getElementById('plain-llm-bleu').innerText = data.calculated_metrics.plain_llm.bleu.toFixed(4);
            document.getElementById('plain-llm-rouge').innerText = data.calculated_metrics.plain_llm.rouge_l.toFixed(4);
            document.getElementById('mongodb-rag-bleu').innerText = data.calculated_metrics.mongodb_rag.bleu.toFixed(4);
            document.getElementById('mongodb-rag-rouge').innerText = data.calculated_metrics.mongodb_rag.rouge_l.toFixed(4);
            document.getElementById('neo4j-kg-rag-bleu').innerText = data.calculated_metrics.neo4j_kg_rag.bleu.toFixed(4);
            document.getElementById('neo4j-kg-rag-rouge').innerText = data.calculated_metrics.neo4j_kg_rag.rouge_l.toFixed(4);
        }

        resetFeedbackForms();
        comparisonResultsContainer.style.display = 'block';

    } catch (error) {
        showError(error.message || "Failed to generate comparison.");
    } finally {
        comparisonLoader.style.display = 'none';
        compareBtn.disabled = false;
        compareBtn.textContent = 'Compare RAG Approaches';
    }
});

function resetFeedbackForms() {
    document.querySelectorAll('.feedback-form').forEach(form => {
        const submitBtn = form.querySelector('.submit-feedback-btn');
        submitBtn.disabled = false;
        submitBtn.textContent = 'Submit Rating';
        form.style.opacity = '1';
        form.querySelectorAll('span, button').forEach(el => el.style.pointerEvents = 'auto');
        form.querySelectorAll('.stars').forEach(container => {
            container.removeAttribute('data-rating');
            container.querySelectorAll('span').forEach(star => star.classList.remove('selected'));
        });
    });
}

document.querySelectorAll('.feedback-form').forEach(form => {
    form.querySelectorAll('.stars').forEach(container => {
        const stars = container.querySelectorAll('span');
        stars.forEach((star, index) => {
            star.addEventListener('click', () => {
                container.dataset.rating = index + 1;
                stars.forEach((s, i) => s.classList.toggle('selected', i <= index));
            });
        });
    });

    const submitBtn = form.querySelector('.submit-feedback-btn');
    submitBtn.addEventListener('click', async () => {
        const modelType = form.dataset.model;
        const ratings = {};
        let allRated = true;
        form.querySelectorAll('.stars').forEach(container => {
            const metric = container.dataset.metric;
            const rating = container.dataset.rating;
            if (rating) {
                ratings[metric] = parseInt(rating, 10);
            } else {
                allRated = false;
            }
        });

        if (!allRated) {
            alert('Please rate all criteria before submitting.');
            return;
        }
        
        submitBtn.textContent = 'Submitting...';
        submitBtn.disabled = true;

        try {
            const response = await fetch('/save_feedback', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    session_id: graphSessionId, model_type: modelType, ratings: ratings,
                    query: document.getElementById('query-input').value.trim()
                })
            });
            const result = await response.json();
            if (response.ok && result.success) {
                submitBtn.textContent = 'Rating Submitted!';
                form.querySelectorAll('span, button').forEach(el => el.style.pointerEvents = 'none');
                form.style.opacity = '0.6';
            } else {
                throw new Error(result.message || 'Failed to save feedback.');
            }
        } catch (error) {
            alert(`Error: ${error.message}`);
            submitBtn.textContent = 'Submit Rating';
            submitBtn.disabled = false;
        }
    });
});