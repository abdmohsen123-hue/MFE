from src.helper import *
from src.Dijkstras import *
import matplotlib.colors as colors
from matplotlib.cm import ScalarMappable
def plot_disctrized_phase_space(Data,Disctrize_box_size=10):
    Nd=Disctrize_box_size
    x=Data.copy()      
    mini = np.min(x, axis=0)        #Extract minimum value of all physical variables 
    maxi = np.max(x, axis=0)        #Extract maximum value of all physical variables
    delta=(maxi+10e-12-mini)/Nd            #Discretize the state space into cells
    idx = np.floor((x - mini) / delta).astype(int)

    grid = np.zeros((Nd, Nd))

    for i in range(len(idx)):
        E_idx = idx[i, 0]
        d_idx = idx[i, 1]
        grid[E_idx,d_idx] += 1   
    # normalize (optional)
    #grid = grid / np.max(grid)
    vmin_1, vmax_1 = np.min(grid), np.max(grid)
    norm_1 = colors.Normalize(vmin_1, vmax_1)
    tick_values_1 = np.linspace(vmin_1, vmax_1, num=5)
    sm1 = ScalarMappable(norm=norm_1, cmap='RdBu_r')

    mask=np.where(grid>0)
    grid_normalized=np.full_like(grid,0)
    grid_normalized[mask] = np.log(grid[mask])
    grid_normalized =grid_normalized/np.max(grid_normalized)
    color_map = cm.RdBu_r(grid_normalized)
    print(color_map.shape)
    color="#000204"


    fig, ax = plt.subplots(dpi=300)
    it =0
    for i in range(Nd):
        for j in range(Nd):
            if grid[i, j] > 0:
                # draw cell edges
                ax.plot([j, j+1], [i, i], linewidth=1,c=color)
                ax.plot([j, j+1], [i+1, i+1], linewidth=1,c=color)
                ax.plot([j, j], [i, i+1], linewidth=1,c=color)
                ax.plot([j+1, j+1], [i, i+1], linewidth=1,c=color)
                
                #ax.text(j + 0.5, i + 0.5, str(int(it)), color='black', ha='center', va='center', fontsize=140/Nd)
                ax.scatter(j + 0.5, i + 0.5, s=3000/Nd, color=color_map[i,j], alpha=0.6,marker="s")
                it+=1
    ax.set_xlabel("Δk",fontsize=14)
    ax.set_ylabel("Δε",fontsize=14)
    #ax.set_title(f"Partitioning of the k-ε space")
    #ax.set_facecolor("#FCFBEA")
    #fig.colorbar(sm1, ax=ax, orientation='vertical', location='right',ticks=tick_values_1)
def plot_box_transition_matrix(CPT,k=-1,sim_time=np.inf,sample_time=1):
    Ap=cluster_transition_matrix(CPT,k,sim=sim_time,sample=sample_time)
    fig_trans = plt.figure(figsize=(10, 8),dpi=500)
    ax_trans = plt.gca()
    im_trans = ax_trans.imshow(Ap, cmap='Blues',norm=mcolors.LogNorm(vmin=1e-6, vmax=0.5))
    ax_trans.set_xlabel('From Box')
    ax_trans.set_ylabel('To Box')
    ax_trans.set_ylim(Ap.shape[0],-1)
    ax_trans.set_xlim(-1, Ap.shape[1])

    ax_trans.set_title('Box Transition Probability Matrix')
    plt.colorbar(im_trans, ax=ax_trans, label='Probability')
    plt.show()

def plot_cluster_centers(CPT,Data,k=-1,threshold=1,Path=''):
    CP=CPT[k]
    N=k
    extremePoints, critical_points = extreme_points(Data, threshold)
    CC=cluster_centroids(CPT,Data,k=N)
    colors,c = cluster_colors(CPT,k=N)
    nclust = len(np.unique(CP))
    plt.figure(figsize=(8, 8))
    plt.scatter(Data[:, 1], Data[:, 0], s=5, c=colors)
    plt.xlabel("Energy")
    plt.ylabel("Dissipation")
    plt.title(f"Cluster Centers, number of clusters: {nclust}")
    for i in range(nclust):
        plt.scatter(CC[i,1], CC[i,0],s=200,facecolors="red")
        plt.text(CC[i,1], CC[i,0], str(i),
            color='white',
            ha='center',
            va='center',
            fontsize=8,
            weight='bold',
        )
    #plt.ylim(Data[:, 0].min() - 0.1, Data[:, 0].max() + 0.1)
    #plt.xlim(Data[:, 1].min() - 0.1, Data[:, 1].max() + 0.1)
    plt.tight_layout()
    plt.axvline(x=critical_points[1], color='red', linestyle='--', label='Critical Point')
    plt.axhline(y=critical_points[0], color='red', linestyle='--', label='Critical Point')
    if Path!="":
        plt.savefig(os.path.join(Path, f"Cluster_Centers_nclust={nclust}.png"))
    plt.show()

def plot_cluster_transition_matrix(CPT,method="CPT",k=-1,sim_time=np.inf,sample_time=1):
    if method=="CPT":
        N=k
        CTM=cluster_transition_matrix(CPT,k=N,sim=sim_time,sample=sample_time)
    elif method=="CTM":
        CTM=CPT
    fig_trans = plt.figure(figsize=(10, 8))
    ax_trans = plt.gca()
    im_trans = ax_trans.imshow(CTM, cmap='YlOrRd',norm=mcolors.LogNorm(vmin=1e-3, vmax=CTM.max()))
    ax_trans.set_xlabel('From Cluster')
    ax_trans.set_ylabel('To Cluster')
    ax_trans.set_title('Cluster Transition Probability Matrix')
    ax_trans.yaxis.set_major_locator(MultipleLocator(1))
    ax_trans.xaxis.set_major_locator(MultipleLocator(1))
    plt.colorbar(im_trans, ax=ax_trans, label='Probability')
    plt.show()

def plot_markov_chain(CPT,k=-1, threshold=0,type="CPT",sim_time=np.inf,sample_time=1):
    def plot_markov_chain_CTM(CTM, threshold=0):
        n = CTM.shape[0]
        G = nx.DiGraph()
        # Add nodes
        for i in range(n):
            G.add_node(i)
        # Add edges
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                p = CTM[j, i]
                if p > threshold:
                    G.add_edge(i, j, weight=p)
        # Layout
        pos = nx.shell_layout(G)
        plt.figure(figsize=(8,8))
        # Node colors
        c = cm.tab20(np.linspace(0, 1, len(G.nodes())))
        nx.draw_networkx_nodes(
            G,
            pos,
            node_size=500,
            node_color=c
        )
        nx.draw_networkx_labels(
            G,
            pos,
            font_size=12
        )
        # Separate edge types
        double_edges = []
        single_edges = []
        for u, v in G.edges():
            # Bidirectional edge
            if G.has_edge(v, u):
                # avoid duplicates
                if u < v:
                    double_edges.append((u, v))
            else:
                single_edges.append((u, v))
        # Draw bidirectional edges
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=double_edges,
            width=2,
            arrows=True,
            arrowstyle='<|-|>',
            arrowsize=20,
            connectionstyle='arc3,rad=0.1'
        )
        # Draw single-direction edges
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=single_edges,
            width=2,
            arrows=True,
            arrowstyle='-|>',
            arrowsize=20,
            connectionstyle='arc3,rad=0.1'
        )
        plt.title("Markov Chain")
        plt.axis('off')
        plt.show()
    N  = k
    CTM = cluster_transition_matrix(CPT,k=N,sim=sim_time,sample=sample_time)
    plot_markov_chain_CTM(CTM, threshold)


def plotModeCoefficients(a, N, dt,colors="black",Path=""):
    plt.rcParams["font.family"] = "DejaVu Serif"
    t = np.linspace(1,N,N)*dt
    npoints =N
    fig1 = plt.figure(figsize=(20, 12),dpi=300)
    for i in range(9):
        match i:
            case 0:                title = "Basic Flow"
            case 1:                title = "Streaks"
            case 2:                title = "Streamwise Vortices"
            case 3:                title = "Spanwise Flow"
            case 4:                title = "Spanwise Flow"
            case 5:                title = "Normal Vertex"
            case 6:                title = "Normal Vertex"
            case 7:                title = "Fully Dimensional Mode"
            case 8:                title = "Modified Basic Flow"
        ax1 = fig1.add_subplot(3,3,i+1)
        ax1.autoscale(enable=True, axis='y', tight=True)
        if colors=="black":
            ax1.plot(t[:npoints], a[:npoints,i],c="black", linewidth=0.5)
        else:
            ax1.scatter(t[:npoints], a[:npoints,i],c=colors, linewidth=0.005)
        ax1.set_xlabel("Time", fontsize=16,fontweight='bold')
        ax1.set_ylabel(f"$a_{{{i+1}}}$", fontsize=14,fontweight='bold')
        ax1.set_title(title, fontsize=18,fontweight='bold')
        ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    if Path!="":
        plt.savefig(os.path.join(Path, f"Mode_Coefficients.png"))
    plt.show()

def plotClusterMembership(Path,CPT, npoints, dt,k=-1):
    CP=CPT[k]
    plt.rcParams["font.family"] = "DejaVu Serif"
    t = np.linspace(1,(npoints),(npoints))*dt
    fig2 = plt.figure(figsize=(12, 6))
    ax2 = fig2.add_subplot(111)
    ax2.scatter(t[:npoints], CP[:npoints], color='black', s=5)
    ax2.set_xlabel("Time", fontsize=14,fontweight='bold')
    ax2.set_ylabel("Cluster Assignment", fontsize=14,fontweight='bold')
    ax2.set_title("Cluster Membership Over Time", fontsize=16,fontweight='bold')
    ax2.yaxis.set_major_locator(MultipleLocator(1))
    ax2.grid(True, alpha=0.3)
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(Path, f"Cluster_Membership.png"))
    plt.show()

def plotDissipationAndEnergy(DE,N,dt,colors="black",Path=""):
    plt.rcParams["font.family"] = "DejaVu Serif"
    t = np.linspace(1,N,N)*dt
    D = DE[:, 0]
    E = DE[:, 1]
    fig1 = plt.figure(figsize=(14, 12))
    npoints = N
    ax1 = fig1.add_subplot(212)
    if colors=="black":
        ax1.plot(t[:npoints], D[:npoints],c="black", linewidth=0.5)
    else:
        ax1.scatter(t[:npoints], D[:npoints],c=colors[:npoints], S=1)
    ax1.set_xlabel("t", fontsize=18,fontweight='bold')
    ax1.set_ylabel("ε", fontsize=18,fontweight='bold')
    ax1.set_title("Dissipation Rate Over Time", fontsize=18,fontweight='bold')
    plt.xticks(fontweight='bold',fontsize=14)
    plt.yticks(fontweight='bold',fontsize=14)
    ax1.set_xlim(0,N*dt)
    ax1.grid(False)
    ax2 = fig1.add_subplot(211)
    if colors=="black":
        ax2.plot(t[:npoints], E[:npoints],c="black", linewidth=0.5)
    else:
        ax2.scatter(t[:npoints], E[:npoints],c=colors[:npoints], S=1)
    ax2.set_xlabel("t", fontsize=18,fontweight='bold')
    ax2.set_ylabel("k", fontsize=18,fontweight='bold')
    ax2.set_title("Energy Over Time", fontsize=18,fontweight='bold')
    ax2.grid(False)
    ax2.set_xlim(0,N*dt)
    plt.xticks(fontweight='bold',fontsize=14)
    plt.yticks(fontweight='bold',fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(Path, f"Dissipation_Energy.png"))
    plt.show()

def plot_dijkstra_shortest_path(CPT,data, start, end,k=-1):
    CP=CPT[k]
    spp=dijkstra_shortest_path(CP,start,end)
    spp=list(spp[0])
    #print(spp)
    CC=cluster_centroids(CP,data)
    colors,ci = cluster_colors(CP)
    nclust = len(np.unique(CP))
    plt.figure(figsize=(12, 8))
    for i in spp:
        plt.scatter(
            data[CP == i, 1],
            data[CP == i, 0],
            s=5,
            color=ci[i],
        )

    for i in range(nclust):
        if np.isin(i,spp): 
            continue
        plt.scatter(
        data[CP == i, 1],
        data[CP == i, 0],
        s=5,
        color="lightgray",
        )
        '''
    for i in spp:
        plt.scatter(CC[i,1], CC[i,0],s=50,facecolors="red")
        plt.text(CC[i,1], CC[i,0], str(i),
            color='white',
            ha='center',
            va='center',
            fontsize=8,
            weight='bold',
        )'''
    for i in spp:
        plt.scatter(CC[i,1], CC[i,0],s=200,facecolors="red")
        plt.text(CC[i,1], CC[i,0], str(i),
            color='white',
            ha='center',
            va='center',
            fontsize=8,
            weight='bold',
        )
    #plt.xlim(0.1,0.2)
    #plt.ylim(0.3,1)
    plt.xlabel("Energy")
    plt.ylabel("Dissipation")
    plt.title(f"Dijkstra's shortest path from cluster {start} to cluster {end}")
    #plt.savefig(os.path.join(Case, f"Dijkstra_shortest_path_{start}_to_{end}.png"))
    plt.show()
    
def plot_cluster_distance_matrix(Path,CPT, Data,k=-1):
    CP=CPT[k]
    N=k
    CC=cluster_centroids(CPT,Data,k=N)
    nclust = len(np.unique(CP))
    DS=np.zeros((nclust, nclust))
    for i in range(nclust):
        for j in range(nclust):
            DS[i, j] = np.linalg.norm(CC[i] - CC[j])
    fig4 = plt.figure(figsize=(8, 6))
    ax = plt.gca()
    im = ax.imshow(DS, cmap='coolwarm')
    # Force integer tick positions
    ax.set_xticks(np.arange(nclust))
    ax.set_yticks(np.arange(nclust))
    # Optional: label them as integers
    ax.set_xticklabels(np.arange(nclust))
    ax.set_yticklabels(np.arange(nclust))
    plt.colorbar(im, ax=ax, label='Distance')
    plt.title('Distance Matrix Between Clusters')
    plt.xlabel('Cluster')
    plt.ylabel('Cluster')
    plt.savefig(os.path.join(Path, f"Distance_Matrix_Clusters.png"))
    plt.show()