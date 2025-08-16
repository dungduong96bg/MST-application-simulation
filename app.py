import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.datasets import load_iris
from scipy.cluster.hierarchy import linkage, fcluster
import pandas as pd
from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

st.set_page_config(page_title="MST Interactive App", layout="wide")

# ===========================
# Tab selection
# ===========================
tab = st.sidebar.radio("Chọn bài toán:", ["Clustering - agglomerative hierarchical clustering", "Bài toán kết nối mạng thành phố"])

# ===========================
# Tab 1: Iris MST + Clustering
# ===========================
if tab == "Clustering - agglomerative hierarchical clustering":
    st.header("Bài toán 1: Clustering - Chiến lược hợp nhât")
    with st.expander("ℹ️ Cây bao trùm và thuật toán phân loại theo chiến lược hợp nhất Agglomerative / Single Linkage Clustering"):
        st.markdown("""
    - **Minimum Spanning Tree (MST)** là đồ thị nối tất cả các điểm với tổng trọng số (khoảng cách) nhỏ nhất mà **không tạo chu trình**.  
    MST giúp xác định cách kết nối các điểm **gần nhau nhất** thông qua các cạnh - khoảng cách ngắn nhất.

    Bài toán phân loại **Agglomerative Hierarchical Clustering - Single Linkage Clustering**:

    - Khoảng cách giữa hai cluster = **khoảng cách ngắn nhất** giữa các điểm thuộc hai cluster.
    - MST cung cấp công cụ trực quan: **loại bỏ (k-1) cạnh dài nhất** trên MST sẽ chia dữ liệu thành **k cluster**.
    - Vì MST đã nối các điểm theo cạnh ngắn nhất, việc loại bỏ các cạnh dài nhất tương ứng với việc **tách các nhóm dữ liệu**.

    **=>** MST giúp thuật toán Agglomerative xác định **cấu trúc phân nhóm** của dữ liệu và hỗ trợ **trực quan hóa cluster**.
    """)

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

    show_mst = st.sidebar.checkbox("Hiển thị MST", value=False)
    cut_k_clusters = st.sidebar.checkbox("Tạo k cụm", value=False   )
    k_clusters = st.sidebar.number_input("Chọn số cụm k", min_value=2, max_value=5, value=3, step=1)

    # Build MST
    G = nx.Graph()
    for i in range(n_points):
        for j in range(i + 1, n_points):
            dist = np.linalg.norm(X_full[i] - X_full[j])
            G.add_edge(i, j, weight=dist)
    MST = nx.minimum_spanning_tree(G, algorithm='kruskal')
    edges_sorted = sorted(MST.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)

    # Clustering
    clusters = [set(range(n_points))]
    edges_to_remove = []
    if cut_k_clusters:
        Z = linkage(X_full, method='single')
        labels = fcluster(Z, t=k_clusters, criterion='maxclust')
        clusters = [set(np.where(labels == i)[0]) for i in range(1, k_clusters + 1)]
        edges_to_remove = edges_sorted[:k_clusters - 1]

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
            st.markdown("### Các cạnh bị loại bỏ")
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
elif tab == "Bài toán kết nối mạng thành phố":
    st.header("Bài toán 2: Kết nối BTS và Nhà bằng MST")

    # --- Giải thích lý thuyết ---
    with st.expander("ℹ️ Bài toán kết nối mạng thành phố"):
        st.markdown("""
        **MST (Minimum Spanning Tree)** trong mạng thành phố giúp:
        - Kết nối tất cả các node (BTS và Nhà) với tổng chiều dài dây nhỏ nhất.
        - Không tạo chu trình → tránh lãng phí kết nối.
        - Tối ưu chi phí và đảm bảo tất cả các nhà đều nhận được tín hiệu từ BTS.

        **Ứng dụng:**
        - Xác định các kết nối cần thiết nhất.
        - Giảm thiểu chi phí thi công.
        """)

    # --- Sidebar cấu hình ---
    st.sidebar.subheader("Cấu hình mạng thành phố")
    n_bts = st.sidebar.number_input("Số BTS", min_value=2, max_value=5, value=2, step=1)
    n_house = st.sidebar.number_input("Số Nhà", min_value=5, max_value=15, value=8, step=1)
    show_mst = st.sidebar.checkbox("Hiển thị MST", value=False)

    # --- Tạo dữ liệu ban đầu ---
    np.random.seed(11021996)


    def generate_positions(n, x_range, y_range, min_dist=15):
        points = []
        while len(points) < n:
            x = np.random.uniform(*x_range)
            y = np.random.uniform(*y_range)
            if all(np.linalg.norm(np.array([x, y]) - np.array(p)) >= min_dist for p in points):
                points.append([x, y])
        return np.array(points)


    # Tạo BTS và Nhà
    points_bts = generate_positions(n_bts, (20, 80), (5,100), min_dist=8)  # BTS ở trên
    points_house = generate_positions(n_house, (10, 90), (5, 100), min_dist=8)  # Nhà ở dưới

    img_bts = Image.open("bts_icons.png")
    img_house = Image.open("house.png")

    # --- Tạo graph ---
    G = nx.Graph()
    for i in range(n_bts):
        G.add_node(f"BTS{i}", type="bts", pos=points_bts[i])
    for j in range(n_house):
        G.add_node(f"H{j}", type="house", pos=points_house[j])

    bts_nodes = [n for n in G.nodes if G.nodes[n]["type"] == "bts"]
    house_nodes = [n for n in G.nodes if G.nodes[n]["type"] == "house"]

    for bts in bts_nodes:
        for house in house_nodes:
            dist = np.linalg.norm(G.nodes[bts]["pos"] - G.nodes[house]["pos"])
            G.add_edge(bts, house, weight=dist)

    # --- MST ---
    MST = nx.minimum_spanning_tree(G, algorithm='kruskal')
    pos = nx.get_node_attributes(G, 'pos')

    # --- Vẽ ---
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    if show_mst:
        nx.draw_networkx_edges(MST, pos, ax=ax, edge_color='gray', width=1)
        edge_labels = {(u, v): f"{d['weight']:.1f}" for u, v, d in MST.edges(data=True)}
        nx.draw_networkx_edge_labels(MST, pos, edge_labels=edge_labels, font_size=7)

    def imscatter(x, y, image, ax=None, zoom=0.08):
        if ax is None:
            ax = plt.gca()
        im = OffsetImage(image, zoom=zoom)
        ab = AnnotationBbox(im, (x, y), frameon=False)
        ax.add_artist(ab)

    for node, (x, y) in pos.items():
        if G.nodes[node]["type"] == "bts":
            imscatter(x, y, img_bts, ax=ax, zoom=0.1)
        else:
            imscatter(x, y, img_house, ax=ax, zoom=0.07)

    col1, col2 = st.columns([3, 2])
    with col1:
        st.pyplot(fig)
    with col2:
        st.markdown("### Thống kê cạnh MST" if show_mst else "### Danh sách node")
        if show_mst:
            edges_data = []
            total_length = 0
            for u, v, d in MST.edges(data=True):
                total_length += d["weight"]
                edges_data.append({
                    "Node 1": u,
                    "Node 2": v,
                    "Khoảng cách": round(d["weight"], 2)
                })
            df_edges = pd.DataFrame(edges_data)
            st.dataframe(df_edges)
            st.markdown(f"**Tổng chiều dài dây:** {total_length:.2f} đơn vị")
        else:
            nodes_info = [{"Node": n, "Loại": G.nodes[n]["type"]} for n in G.nodes]
            st.dataframe(pd.DataFrame(nodes_info))

    # --- Kết luận ---
    if show_mst:
        st.markdown("""
        **Kết luận:**  
        - MST giúp kết nối tất cả BTS và nhà với tổng chiều dài dây ngắn nhất.  
        - Đây là phương án tối ưu để triển khai mạng lưới với chi phí thấp nhất mà vẫn đảm bảo kết nối.
        """)
    else:
        st.markdown("""
        **Kết luận:**  
        - Khi không sử dụng MST, bản đồ chỉ hiển thị vị trí các node.  
        - Không có thông tin về kết nối tối ưu → không xác định được chi phí tối thiểu.
        """)

