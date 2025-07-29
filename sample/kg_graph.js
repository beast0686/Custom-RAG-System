// Global state management
const AppState = {
    knowledgeGraph: null,
    currentGraphData: null,
    allNodesData: [],
    filteredData: { nodes: [], links: [] }
};

// Initialize application
document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
});

function initializeApp() {
    setupEventListeners();
    addLog('system', 'API Comparison Interface initialized');
}

function setupEventListeners() {
    // Modal events
    const modal = document.getElementById('node-details-modal');
    const closeModal = document.querySelector('.close-modal');
    
    if (closeModal) {
        closeModal.onclick = () => modal.style.display = 'none';
    }
    
    // Close modal on outside click
    window.onclick = (event) => {
        if (event.target === modal) {
            modal.style.display = 'none';
        }
    };
}

// API Communication
async function sendRequest() {
    const userInput = document.getElementById("user_input").value.trim();
    const userId = document.getElementById("user_id_input").value.trim();
    const blockId = document.getElementById("block_id_input").value.trim();
    const select = document.getElementById('api-function-select');

    if (!userInput) {
        addLog('error', 'User input is required');
        return;
    }

    addLog('info', 'Starting API comparison...');

    const prettyName = select.value.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()); // generate_title -> Generate Title
    fullUserPrompt = userInput +  `, Please generate the response by calling this function: ${select.value} - ${prettyName}`
    fullUserPromptForGPT = userInput +`, "${prettyName} for this. Use a confident, witty, and professional tone."`
    
    console.log(fullUserPrompt)
    console.log(fullUserPromptForGPT)

    const payloads = {
        new: {
            user_unique_identifier: userId,
            block_unique_identifier: blockId,
            user_input: fullUserPrompt
        },
        legacy: {
            user_unique_identifier: userId,
            block_unique_identifier: blockId,
            user_input: fullUserPrompt
        },
        chatgpt_answer: {
            question: fullUserPromptForGPT,
        }
    };

    resetUI();
    hideKnowledgeGraph();

    try {
        await Promise.all([
            callNewAPI(payloads.new),
            callLegacyAPI(payloads.legacy),
            callOpenAIAPI(payloads.chatgpt_answer)
        ]);
        
        addLog('success', 'API comparison completed successfully');
    } catch (error) {
        handleAPIError(error);
    } finally {
        hideLoadingOverlay();
    }
}

async function callNewAPI(payload) {
    updateLoadingMessage( `Calling /${CONFIG.cdkg_endpoint}...`);
    addLog('info', `Calling /${CONFIG.cdkg_endpoint}...`);
    
    const response = await fetch(`${CONFIG.API_BASE_URL}/${CONFIG.cdkg_endpoint}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
    });

    const data = await response.json();
    
    document.getElementById("full_json").textContent = JSON.stringify(data, null, 2);
    formatResultContent(data, 'response_1');
    addLog('success', 'New API call completed successfully');

    // Handle Knowledge Graph data
    if (data.kg_data) {
        addLog('info', 'Knowledge Graph data received, initializing visualization...');
        await initializeKnowledgeGraph(data.kg_data);
    }
}

async function callLegacyAPI(payload) {
    updateLoadingMessage(`Calling /${CONFIG.old_endpoint}...`);
    addLog('info', `Calling /${CONFIG.old_endpoint}...`);
    
    const response = await fetch(`${CONFIG.API_BASE_URL}/${CONFIG.old_endpoint}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
    });

    const data = await response.json();
    
    document.getElementById("full_json_2").textContent = JSON.stringify(data, null, 2);

    formatResultContent(data, 'response_2');
    addLog('success', 'Legacy API call completed successfully');
}

async function callOpenAIAPI(payload) {
    updateLoadingMessage(`Calling /${CONFIG.chatgpt_answer}...`);
    addLog('info', `Calling /${CONFIG.chatgpt_answer}...`)

    try {
        const response = await fetch(`${CONFIG.API_BASE_URL}/${CONFIG.chatgpt_answer}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        const data = await response.text();

        document.getElementById("full_json_3").textContent = data;
        
        formatOpenAIResultContent(data, 'response_3');
        addLog('success', 'OpenAI API call completed successfully');
    } catch (error) {
        updateStatusIndicator(document.getElementById("status_3"), 'error', 'Failed');
        document.getElementById("response_3").innerHTML = createResultSection('Error', `OpenAI API call failed: ${error.message}`);
        document.getElementById("full_json_3").textContent = `Error: ${error.message}`;
        addLog('error', `OpenAI API call failed: ${error.message}`);
    }
}

// Knowledge Graph Management
async function initializeKnowledgeGraph(kgData) {
    if (!kgData?.graph_data) {
        addLog('warning', 'No graph data available for Knowledge Graph');
        return;
    }

    try {
        showKnowledgeGraphSection();
        
        AppState.currentGraphData = kgData.graph_data;
        AppState.allNodesData = AppState.currentGraphData.nodes || [];
        AppState.filteredData = { 
            nodes: [...AppState.allNodesData], 
            links: [...(AppState.currentGraphData.links || [])] 
        };
        
        populateFilterOptions();
        await renderKnowledgeGraph(AppState.filteredData);
        
        if (kgData.graph_insights) {
            displayGraphInsights(kgData.graph_insights);
        }
        
        addLog('success', `Knowledge Graph initialized with ${AppState.filteredData.nodes.length} nodes and ${AppState.filteredData.links.length} links`);
        
    } catch (error) {
        addLog('error', `Failed to initialize Knowledge Graph: ${error.message}`);
    }
}

async function renderKnowledgeGraph(data) {
    const graphContainer = document.getElementById("graph-container");
    graphContainer.innerHTML = '';
    
    try {
        AppState.knowledgeGraph = ForceGraph()
            (graphContainer)
            .graphData(data)
            .nodeId('id')
            .nodeLabel(createNodeLabel)
            .nodeColor(node => node.color || getNodeColor(node.type))
            .nodeVal(node => node.size || getNodeSize(node))
            .linkLabel(link => `${link.type || 'related'} (weight: ${link.weight || 1})`)
            .linkColor(() => 'rgba(100, 100, 100, 0.6)')
            .linkWidth(link => Math.sqrt(link.weight || 1) * 2)
            .onNodeClick(showNodeDetails)
            .d3Force('charge', d3.forceManyBody().strength(-300))
            .d3Force('link', d3.forceLink().distance(100))
            .cooldownTicks(100)
            .warmupTicks(100);
            
        // Auto-fit graph
        setTimeout(() => {
            if (AppState.knowledgeGraph) {
                AppState.knowledgeGraph.zoomToFit(400);
            }
        }, 2000);
        
        addLog('success', 'Knowledge Graph rendered successfully');
        
    } catch (error) {
        graphContainer.innerHTML = `
            <div class="loading-placeholder">
                <p style="color: #dc2626;">Error rendering Knowledge Graph: ${error.message}</p>
            </div>
        `;
        addLog('error', `Graph rendering failed: ${error.message}`);
    }
}

// Node and Link Management
function createNodeLabel(node) {
    return `
        <div style="background: rgba(0,0,0,0.8); color: white; padding: 8px; border-radius: 4px; max-width: 200px;">
            <strong>${node.title || 'Unknown'}</strong><br/>
            <em>Type: ${node.type || 'N/A'}</em><br/>
            <em>Domain: ${node.domain || 'N/A'}</em>
            ${node.similarity_score ? `<br/><em>Relevance: ${(node.similarity_score * 100).toFixed(1)}%</em>` : ''}
        </div>
    `;
}

function getNodeColor(type) {
    return CONFIG.NODE_COLORS[type] || CONFIG.NODE_COLORS.unknown;
}

function getNodeSize(node) {
    let size = CONFIG.NODE_SIZES[node.type] || CONFIG.NODE_SIZES.unknown;
    
    if (node.type === 'knowledge' && node.similarity_score) {
        size += node.similarity_score * 8;
    }
    
    return size;
}

function showNodeDetails(node) {
    const modal = document.getElementById('node-details-modal');
    const title = document.getElementById('node-details-title');
    const body = document.getElementById('node-details-body');
    
    title.textContent = node.title || 'Unknown Node';
    body.innerHTML = generateNodeDetailsHTML(node);
    modal.style.display = 'block';
    modal.classList.add('fade-in');
    
    addLog('info', `Opened details for node: ${node.title || node.id}`);
}

function generateNodeDetailsHTML(node) {
    let html = `
        <div class="node-detail-section">
            <h5>Basic Information</h5>
            <p><strong>Type:</strong> ${escapeHtml(node.type || 'N/A')}</p>
            <p><strong>Domain:</strong> ${escapeHtml(node.domain || 'N/A')}</p>
            <p><strong>ID:</strong> ${escapeHtml(node.id || 'N/A')}</p>
    `;
    
    // Add additional properties
    const additionalProps = ['similarity_score', 'knowledge_type', 'publication_date'];
    additionalProps.forEach(prop => {
        if (node[prop] !== undefined && node[prop] !== null) {
            const label = prop.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
            let value = node[prop];
            
            if (prop === 'similarity_score') {
                value = `${(value * 100).toFixed(1)}%`;
            }
            
            html += `<p><strong>${label}:</strong> ${escapeHtml(String(value))}</p>`;
        }
    });
    
    html += `</div>`;
    
    // Add connected nodes
    if (AppState.currentGraphData?.links) {
        const connectedLinks = AppState.currentGraphData.links.filter(link => 
            link.source.id === node.id || link.target.id === node.id
        );
        
        if (connectedLinks.length > 0) {
            html += generateConnectedNodesHTML(node, connectedLinks);
        }
    }
    
    // Add raw data if substantial
    if (node.data && typeof node.data === 'object' && Object.keys(node.data).length > 0) {
        html += `
            <div class="node-detail-section">
                <h5>Additional Data</h5>
                <pre class="node-raw-data">${JSON.stringify(node.data, null, 2)}</pre>
            </div>
        `;
    }
    
    return html;
}

function generateConnectedNodesHTML(node, connectedLinks) {
    let html = `
        <div class="node-detail-section">
            <h5>Connected Entities (${connectedLinks.length})</h5>
            <div class="connected-nodes">
    `;
    
    connectedLinks.slice(0, 10).forEach(link => {
        const connectedNode = link.source.id === node.id ? link.target : link.source;
        html += `
            <div class="connected-node">
                <span class="connection-type">${escapeHtml(link.type || 'related')}</span>
                <strong>${escapeHtml(connectedNode.title || 'Unknown')}</strong>
                <em>(${escapeHtml(connectedNode.type || 'N/A')})</em>
                ${link.weight ? `<span style="float: right; color: var(--text-muted);">Weight: ${link.weight.toFixed(2)}</span>` : ''}
            </div>
        `;
    });
    
    if (connectedLinks.length > 10) {
        html += `<p style="text-align: center; color: var(--text-muted); font-style: italic;">... and ${connectedLinks.length - 10} more connections</p>`;
    }
    
    html += `</div></div>`;
    return html;
}

// Filter and Search Functions
function searchNodes() {
    const searchTerm = document.getElementById('node-search').value.toLowerCase().trim();
    
    if (!searchTerm) {
        clearSearch();
        return;
    }
    
    const matchingNodes = AppState.allNodesData.filter(node => 
        (node.title && node.title.toLowerCase().includes(searchTerm)) ||
        (node.domain && node.domain.toLowerCase().includes(searchTerm)) ||
        (node.type && node.type.toLowerCase().includes(searchTerm)) ||
        (node.id && node.id.toLowerCase().includes(searchTerm))
    );
    
    updateFilteredData(matchingNodes);
    addLog('info', `Search found ${matchingNodes.length} matching nodes`);
}

function clearSearch() {
    document.getElementById('node-search').value = '';
    document.getElementById('domain-filter').value = 'all';
    document.getElementById('node-type-filter').value = 'all';
    
    AppState.filteredData = { 
        nodes: [...AppState.allNodesData], 
        links: [...AppState.currentGraphData.links] 
    };
    
    refreshViews();
    addLog('info', 'Search and filters cleared, showing all nodes');
}

function filterByDomain(domain) {
    const searchTerm = document.getElementById('node-search').value.toLowerCase().trim();
    const nodeType = document.getElementById('node-type-filter').value;
    
    let nodes = [...AppState.allNodesData];
    
    if (domain !== 'all') {
        nodes = nodes.filter(node => node.domain === domain);
    }
    
    if (searchTerm) {
        nodes = nodes.filter(node => 
            (node.title && node.title.toLowerCase().includes(searchTerm)) ||
            (node.domain && node.domain.toLowerCase().includes(searchTerm)) ||
            (node.type && node.type.toLowerCase().includes(searchTerm)) ||
            (node.id && node.id.toLowerCase().includes(searchTerm))
        );
    }
    
    if (nodeType !== 'all') {
        nodes = nodes.filter(node => node.type === nodeType);
    }
    
    updateFilteredData(nodes);
    addLog('info', domain === 'all' ? 'Domain filter cleared' : `Filtered to domain "${domain}": ${nodes.length} nodes`);
}

function filterByNodeType(nodeType) {
    const searchTerm = document.getElementById('node-search').value.toLowerCase().trim();
    const domain = document.getElementById('domain-filter').value;
    
    let nodes = [...AppState.allNodesData];
    
    if (nodeType !== 'all') {
        nodes = nodes.filter(node => node.type === nodeType);
    }
    
    if (domain !== 'all') {
        nodes = nodes.filter(node => node.domain === domain);
    }
    
    if (searchTerm) {
        nodes = nodes.filter(node => 
            (node.title && node.title.toLowerCase().includes(searchTerm)) ||
            (node.domain && node.domain.toLowerCase().includes(searchTerm)) ||
            (node.type && node.type.toLowerCase().includes(searchTerm)) ||
            (node.id && node.id.toLowerCase().includes(searchTerm))
        );
    }
    
    updateFilteredData(nodes);
    addLog('info', nodeType === 'all' ? 'Node type filter cleared' : `Filtered to type "${nodeType}": ${nodes.length} nodes`);
}

function updateFilteredData(nodes) {
    const nodeIds = new Set(nodes.map(node => node.id));
    const relevantLinks = AppState.currentGraphData.links.filter(link =>
        nodeIds.has(link.source.id) && nodeIds.has(link.target.id)
    );
    
    AppState.filteredData = { nodes, links: relevantLinks };
    refreshViews();
}

function refreshViews() {
    // Re-render graph
    if (AppState.knowledgeGraph) {
        renderKnowledgeGraph(AppState.filteredData);
    }
    
    // Update cards view if active
    if (document.getElementById('data-cards-container').style.display !== 'none') {
        generateDataCards(AppState.filteredData.nodes);
    }
}

// View Management
function switchView(viewType) {
    const graphContainer = document.getElementById('graph-container');
    const cardsContainer = document.getElementById('data-cards-container');
    const viewButtons = document.querySelectorAll('.view-btn');
    
    // Update button states
    viewButtons.forEach(btn => {
        btn.classList.toggle('active', btn.dataset.view === viewType);
    });
    
    if (viewType === 'graph') {
        graphContainer.style.display = 'block';
        cardsContainer.style.display = 'none';
        
        if (AppState.knowledgeGraph && AppState.filteredData) {
            renderKnowledgeGraph(AppState.filteredData);
        }
        addLog('info', 'Switched to graph view');
    } else if (viewType === 'cards') {
        graphContainer.style.display = 'none';
        cardsContainer.style.display = 'block';
        
        generateDataCards(AppState.filteredData.nodes);
        addLog('info', 'Switched to cards view');
    }
}

function generateDataCards(nodes) {
    const cardsContainer = document.getElementById('data-cards-container');
    
    if (!nodes || nodes.length === 0) {
        cardsContainer.innerHTML = `
            <div class="loading-placeholder">
                <p>No nodes to display</p>
            </div>
        `;
        return;
    }
    
    const cardsHTML = nodes.map(node => `
        <div class="data-card" onclick='showNodeDetails(${JSON.stringify(node).replace(/'/g, "&#39;")})'>
            <div class="card-header">
                <div class="card-icon ${node.type || 'unknown'}">
                    ${getNodeTypeIcon(node.type)}
                </div>
                <div class="card-title">${escapeHtml(node.title || 'Unknown')}</div>
            </div>
            <div class="card-details">
                <p><strong>Type:</strong> ${escapeHtml(node.type || 'N/A')}</p>
                <p><strong>Domain:</strong> ${escapeHtml(node.domain || 'N/A')}</p>
                ${node.similarity_score ? `<p><strong>Relevance:</strong> ${(node.similarity_score * 100).toFixed(1)}%</p>` : ''}
                ${node.knowledge_type ? `<p><strong>Knowledge Type:</strong> ${escapeHtml(node.knowledge_type)}</p>` : ''}
                ${node.publication_date ? `<p><strong>Publication:</strong> ${escapeHtml(node.publication_date)}</p>` : ''}
            </div>
        </div>
    `).join('');
    
    cardsContainer.innerHTML = cardsHTML;
}

function getNodeTypeIcon(type) {
    const icons = {
        knowledge: 'K',
        assignee: 'A', 
        keyword: '#',
        technology: 'T',
        unknown: '?'
    };
    return icons[type] || icons.unknown;
}

// UI Helper Functions
function populateFilterOptions() {
    const domainFilter = document.getElementById('domain-filter');
    const domains = [...new Set(AppState.allNodesData.map(node => node.domain).filter(Boolean))];
    
    domainFilter.innerHTML = '<option value="all">All Domains</option>';
    domains.forEach(domain => {
        const option = document.createElement('option');
        option.value = domain;
        option.textContent = domain;
        domainFilter.appendChild(option);
    });
}

function displayGraphInsights(insights) {
    const insightsContainer = document.getElementById('insights-content');
    
    if (!insights || Object.keys(insights).length === 0) {
        insightsContainer.innerHTML = '<div class="loading-placeholder"><p>No insights available</p></div>';
        return;
    }
    
    let insightsHTML = '';
    
    // Central entities
    if (insights.central_entities?.length > 0) {
        insightsHTML += `
            <div class="insight-item">
                <div class="insight-title">Most Central Entities</div>
                <div class="insight-content">
                    ${insights.central_entities.slice(0, 5).map(entity => 
                        `<p><strong>${entity.title || 'Unknown'}</strong> (${entity.type || 'N/A'}) - 
                        Centrality: ${entity.centrality.toFixed(3)}, Connections: ${entity.degree}</p>`
                    ).join('')}
                </div>
            </div>
        `;
    }
    
    // Domain connections
    if (insights.domain_connections?.length > 0) {
        insightsHTML += `
            <div class="insight-item">
                <div class="insight-title">Cross-Domain Connections</div>
                <div class="insight-content">
                    ${insights.domain_connections.slice(0, 5).map(connection => 
                        `<p><strong>${connection.from_domain}</strong> ↔ <strong>${connection.to_domain}</strong><br/>
                        <em>${connection.from_title} → ${connection.to_title}</em></p>`
                    ).join('')}
                </div>
            </div>
        `;
    }
    
    // Graph statistics
    if (insights.total_nodes) {
        insightsHTML += `
            <div class="insight-item">
                <div class="insight-title">Graph Statistics</div>
                <div class="insight-content">
                    <p><strong>Total Nodes:</strong> ${insights.total_nodes}</p>
                    <p><strong>Total Edges:</strong> ${insights.total_edges}</p>
                    <p><strong>Unique Domains:</strong> ${insights.unique_domains}</p>
                </div>
            </div>
        `;
    }
    
    insightsContainer.innerHTML = insightsHTML || '<div class="loading-placeholder"><p>No insights available</p></div>';
}

function formatResultContent(data, containerId) {
    const container = document.getElementById(containerId);
    const status = document.getElementById(containerId.replace('response_', 'status_'));

    const result = data?.FUNCTION_CALL_RESULT?.result || {};
    const message = data?.MESSAGE || {};
    const suggestion = data?.SUGGESTION?.text?.trim?.();
    const summary = data?.summary || {};
    // const insights = data?.insights || {};

    const hasData = Object.keys(result).length || Object.keys(message).length || suggestion || Object.keys(summary).length ;
    //  || Object.keys(insights).length;

    if (!hasData) {
        container.innerHTML = createResultSection('No Data', 'No response data found.');
        updateStatusIndicator(status, 'error', 'No Data');
        return;
    }

    updateStatusIndicator(status, 'success', 'Complete');

    const blocks = [
        ['FUNCTION_CALL_RESULT', result],
        ['MESSAGE', message],
        ['summary', summary],
        // ['INSIGHTS', insights],
        ['SUGGESTION', suggestion]
    ];

    container.innerHTML = blocks.map(([title, content]) => {
        if (!content || (typeof content === 'object' && !Object.keys(content).length)) return '';
        if (typeof content === 'object') {
            return Object.entries(content).map(([k, v]) => formatResultSection(k, v)).join('');
        }
        return formatResultSection(title, content);
    }).join('');
}

function formatOpenAIResultContent(data, containerId) {
    const container = document.getElementById(containerId);
    const status = document.getElementById(containerId.replace('response_', 'status_'));

    if (!data || (typeof data === 'string' && data.trim() === '')) {
        container.innerHTML = createResultSection('No Data', 'No response data found from OpenAI.');
        updateStatusIndicator(status, 'error', 'No Data');
        return;
    }

    updateStatusIndicator(status, 'success', 'Complete');

    // Format the content properly
    container.innerHTML = formatResultSection('OpenAI Response', data);
}

function formatResultSection(title, content) {
    const sectionTitle = title.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
    
    return `
        <div class="result-section">
            <div class="result-section-title">${sectionTitle}</div>
            <div class="result-section-content">${formatContent(content)}</div>
        </div>
    `;
}

function formatContent(content) {
    if (content === null || content === undefined) {
        return '<em>No data available</em>';
    }
    
    if (typeof content === 'string') {
        return formatStringContent(content);
    }
    
    if (Array.isArray(content)) {
        return content.length === 0 ? '<em>Empty list</em>' : 
            `<ul class="result-list">${content.map(item => `<li>${formatContent(item)}</li>`).join('')}</ul>`;
    }
    
    if (typeof content === 'object') {
        const keys = Object.keys(content);
        return keys.length === 0 ? '<em>Empty object</em>' : 
            keys.map(key => `
                <div class="result-object">
                    <div class="result-key">${key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}:</div>
                    <div class="result-value">${formatContent(content[key])}</div>
                </div>
            `).join('');
    }
    
    return String(content);
}

function formatStringContent(str) {
    if (str.includes('\n') && (str.match(/^\d+\./m) || str.match(/^[-•*]/m))) {
        const lines = str.split('\n').filter(line => line.trim());
        return `<ul class="result-list">${lines.map(line => {
            const cleaned = line.trim().replace(/^\d+\.\s*|^[-•*]\s*/, '');
            return cleaned ? `<li>${cleaned}</li>` : '';
        }).filter(Boolean).join('')}</ul>`;
    }
    
    return str.split('\n\n').map(paragraph => 
        `<p>${paragraph.replace(/\n/g, '<br>')}</p>`
    ).join('');
}

function createResultSection(title, content) {
    return `
        <div class="result-section">
            <div class="result-section-title">${title}</div>
            <div class="result-section-content">${content}</div>
        </div>
    `;
}

// UI State Management
function resetUI() {
    updateStatusIndicator(document.getElementById("status_1"), 'loading', 'Loading...');
    updateStatusIndicator(document.getElementById("status_2"), 'loading', 'Loading...');
    updateStatusIndicator(document.getElementById("status_3"), 'loading', 'Loading...');
    
    document.getElementById("response_1").innerHTML = createResultSection('Loading', 'Processing request...');
    document.getElementById("response_2").innerHTML = createResultSection('Loading', 'Processing request...');
    document.getElementById("response_3").innerHTML = createResultSection('Loading', 'Processing request...');
    document.getElementById("full_json").textContent = "Loading...";
    document.getElementById("full_json_2").textContent = "Loading...";
    document.getElementById("full_json_3").textContent = "Loading...";
}

function updateStatusIndicator(indicator, status, text) {
    if (!indicator) return;
    
    indicator.className = `status-indicator ${status}`;
    indicator.innerHTML = `
        <i class="fas ${CONFIG.STATUS_ICONS[status] || CONFIG.STATUS_ICONS.waiting}"></i>
        ${text}
    `;
}

function hideLoadingOverlay() {
    const overlay = document.getElementById('loading-overlay');
    if (overlay) overlay.style.display = 'none';
}

function updateLoadingMessage(message) {
    const messageEl = document.getElementById('loading-message');
    if (messageEl) messageEl.textContent = message;
}

function showKnowledgeGraphSection() {
    const kgSection = document.getElementById("kg-section");
    if (kgSection) kgSection.style.display = "block";
}

function hideKnowledgeGraph() {
    const kgSection = document.getElementById("kg-section");
    if (kgSection) kgSection.style.display = "none";
}

// Error Handling
function handleAPIError(error) {
    const errorMsg = `Error: ${error.message}`;
    
    updateStatusIndicator(document.getElementById("status_1"), 'error', 'Failed');
    updateStatusIndicator(document.getElementById("status_2"), 'error', 'Failed');
    updateStatusIndicator(document.getElementById("status_3"), 'error', 'Failed');
    
    document.getElementById("response_1").innerHTML = createResultSection('Error', errorMsg);
    document.getElementById("response_2").innerHTML = createResultSection('Error', errorMsg);
    document.getElementById("response_3").innerHTML = createResultSection('Error', errorMsg);
    document.getElementById("full_json").textContent = errorMsg;
    document.getElementById("full_json_2").textContent = errorMsg;
    document.getElementById("full_json_3").textContent = errorMsg;
    
    addLog('error', `API call failed: ${error.message}`);
}

// Utility Functions
function addLog(type, message) {
    const consoleLog = document.getElementById('console-log');
    const timestamp = new Date().toLocaleTimeString();
    
    const logElement = document.createElement('div');
    logElement.className = `log-entry ${type}`;
    logElement.innerHTML = `
        <span class="log-time">${timestamp}</span>
        <span class="log-message">${message}</span>
    `;
    
    consoleLog.appendChild(logElement);
    consoleLog.scrollTop = consoleLog.scrollHeight;
}

function clearLogs() {
    const consoleLog = document.getElementById('console-log');
    consoleLog.innerHTML = `
        <div class="log-entry system">
            <span class="log-time">System</span>
            <span class="log-message">Logs cleared</span>
        </div>
    `;
}

function escapeHtml(text) {
    const map = {
        '&': '&amp;', '<': '&lt;', '>': '&gt;',
        '"': '&quot;', "'": '&#39;'
    };
    return String(text).replace(/[&<>"']/g, m => map[m]);
}

function showTemporaryMessage(message) {
    const toast = document.createElement('div');
    toast.className = 'toast-message';
    toast.textContent = message;
    toast.style.cssText = `
        position: fixed; top: 20px; right: 20px;
        background: var(--success-color); color: white;
        padding: 12px 20px; border-radius: 8px; z-index: 10001;
        animation: slideIn 0.3s ease-out;
    `;
    
    document.body.appendChild(toast);
    
    setTimeout(() => {
        toast.style.animation = 'slideOut 0.3s ease-in forwards';
        setTimeout(() => toast.remove(), 300);
    }, 2000);
}

function copyJsonContent(elementId) {
    const element = document.getElementById(elementId);
    if (element) {
        navigator.clipboard.writeText(element.textContent).then(() => {
            addLog('info', 'JSON content copied to clipboard');
            showTemporaryMessage('JSON copied!');
        });
    }
}

function toggleJsonSection(sectionId) {
    const section = document.getElementById(sectionId);
    const button = event.target.closest('.collapse-btn');
    const icon = button.querySelector('i');
    
    if (section.classList.contains('collapsed')) {
        section.classList.remove('collapsed');
        icon.className = 'fas fa-chevron-up';
        addLog('info', 'JSON section expanded');
    } else {
        section.classList.add('collapsed');
        icon.className = 'fas fa-chevron-down';
        addLog('info', 'JSON section collapsed');
    }
}

function toggleConsole() {
    const consoleLog = document.getElementById('console-log');
    const button = event.target.closest('.toggle-console-btn');
    const icon = button.querySelector('i');
    
    if (consoleLog.classList.contains('collapsed')) {
        consoleLog.classList.remove('collapsed');
        icon.className = 'fas fa-chevron-up';
        addLog('info', 'Console expanded');
    } else {
        consoleLog.classList.add('collapsed');
        icon.className = 'fas fa-chevron-down';
        addLog('info', 'Console collapsed');
    }
}

function closeModal() {
    const modal = document.getElementById('node-details-modal');
    modal.style.display = 'none';
}

function showRelatedNodes() {
    addLog('info', 'Show related nodes functionality - highlight related nodes in graph');
    // Could be implemented to highlight related nodes in the graph
}

// Add CSS animations for toast messages
const toastCSS = `
    @keyframes slideIn {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes slideOut {
        from { transform: translateX(0); opacity: 1; }
        to { transform: translateX(100%); opacity: 0; }
    }
`;

// Inject CSS for toast animations
const styleSheet = document.createElement('style');
styleSheet.textContent = toastCSS;
document.head.appendChild(styleSheet);