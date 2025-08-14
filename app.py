import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.datasets import load_iris
from scipy.cluster.hierarchy import linkage, fcluster
import pandas as pd

st.set_page_config(page_title="MST Interactive App", layout="wide")

# ===========================
# Tab selection
# ===========================
tab = st.sidebar.radio("Chọn bài toán:", ["Iris MST + Clustering", "Network thành phố MST"])

# ===========================
# Tab 1: Iris MST + Clustering
# ===========================
if tab == "Iris MST + Clustering":
    st.header("Bài toán 1: MST + Single Linkage Clustering trên Iris")

    # Load Iris
    iris = load_iris()
    X_full = iris.data
    feature_names = iris.feature_names
    n_points = X_full.shape[0]

    # Sidebar config
    st.sidebar.subheader("Cấu hình Iris MST")
    f1 = st.sidebar.selectbox("Chiều X", range(X_full.shape[1]), format_func=lambda i: feature_names[i])
    f2 = st.sidebar.selectbox("Chiều Y", range(X_full.shape[1]), index=1, format_func=lambda i: feature_names[i])
    X_plot = X_full[:, [f1, f2]]

    show_mst = st.sidebar.checkbox("Hiển thị MST", value=True)
    cut_k_clusters = st.sidebar.checkbox("Tạo k cụm", value=True)
    k_clusters = st.sidebar.slider("Chọn số cụm k", 2, 10, 3)

    # Build MST
    G = nx.Graph()
    for i in range(n_points):
        for j in range(i + 1, n_points):
            dist = np.linalg.norm(X_full[i] - X_full[j])
            G.add_edge(i, j, weight=dist)
    MST = nx.minimum_spanning_tree(G, algorithm='kruskal')
    edges_sorted = sorted(MST.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)

    # Single Linkage Clustering
    clusters = [set(range(n_points))]
    edges_to_remove = []
    if cut_k_clusters:
        Z = linkage(X_full, method='single')
        labels = fcluster(Z, t=k_clusters, criterion='maxclust')
        clusters = [set(np.where(labels == i)[0]) for i in range(1, k_clusters + 1)]
        edges_to_remove = edges_sorted[:k_clusters - 1]

    # Cluster colors
    cluster_colors = np.zeros(n_points)
    for idx, comp in enumerate(clusters):
        for node in comp:
            cluster_colors[node] = idx

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    pos = {i: X_plot[i] for i in range(n_points)}
    if show_mst:
        nx.draw_networkx_edges(MST, pos, ax=ax, edge_color="gray", width=2, style="solid")
    for u, v, _ in edges_to_remove:
        ax.plot([X_plot[u, 0], X_plot[v, 0]], [X_plot[u, 1], X_plot[v, 1]], color='red', linewidth=3)
    ax.scatter(X_plot[:, 0], X_plot[:, 1], c=cluster_colors, cmap=plt.cm.Set1, s=50, edgecolors='k')
    ax.set_xlabel(feature_names[f1])
    ax.set_ylabel(feature_names[f2])
    ax.set_title(f"MST + Single Linkage Clustering: k={k_clusters}")
    ax.axis('equal')
    plt.tight_layout()

    # Layout: plot + tables
    col1, col2 = st.columns([3, 2])
    with col1:
        st.pyplot(fig)
    with col2:
        st.markdown("### Thống kê các cụm")
        cluster_info = []
        for idx, comp in enumerate(clusters):
            cluster_info.append({
                "Nhóm": f"Nhóm {idx + 1}",
                "Số quan sát": len(comp),
                "Các điểm": sorted(list(comp))
            })
        st.dataframe(pd.DataFrame(cluster_info))

        if cut_k_clusters:
            st.markdown("### Các cạnh highlight")
            edges_info = []
            for u, v, d in edges_to_remove:
                edges_info.append({
                    "Node 1": u,
                    "Node 2": v,
                    "Trọng số": round(d['weight'], 3)
                })
            st.dataframe(pd.DataFrame(edges_info))

# ===========================
# Tab 2: Network thành phố MST
# ===========================
elif tab == "Network thành phố MST":
    st.header("Bài toán 2: MST ứng dụng mạng thành phố")

    n_points = st.slider("Số điểm trong thành phố", 5, 15, 8)

    # Random points mỗi lần chạy
    points = np.random.rand(n_points, 2) * 100

    # Build MST
    G = nx.Graph()
    for i in range(n_points):
        for j in range(i + 1, n_points):
            dist = np.linalg.norm(points[i] - points[j])
            G.add_edge(i, j, weight=dist)
    MST = nx.minimum_spanning_tree(G, algorithm='kruskal')

    # Chỉnh pos để fit khung
    # points: numpy array (n_points,2)
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    padding = 5

    pos = {i: ((points[i, 0] - x_min) / (x_max - x_min) * (x_max - x_min + 2 * padding) - padding,
               (points[i, 1] - y_min) / (y_max - y_min) * (y_max - y_min + 2 * padding) - padding)
           for i in range(n_points)}

    # Vẽ MST
    fig, ax = plt.subplots(figsize=(3, 3))
    nx.draw_networkx_nodes(MST, pos, ax=ax, node_size=100, node_color='skyblue')
    nx.draw_networkx_edges(MST, pos, ax=ax, edge_color='gray', width=1)
    nx.draw_networkx_labels(MST, pos, ax=ax, font_size=3)

    ax.set_xlim(-padding, x_max - x_min + padding)
    ax.set_ylim(-padding, y_max - y_min + padding)
    ax.set_aspect('equal')
    ax.axis('off')
    st.pyplot(fig)

