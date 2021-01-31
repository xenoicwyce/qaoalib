import warnings
import networkx as nx


def get_info(filename, indicator, mode='get-numeric'):
    try:
        pos = filename.index(indicator)
    except ValueError:
        return None

    start = pos + len(indicator)
    end = start + 1
    if end > len(filename):
        # reached the end of file name
        return filename[start:end]

    if mode == 'get-alpha':
        while filename[start:end].isalpha():
            if end > len(filename):
                break
            end += 1
    else:
        while filename[start:end].isnumeric():
            if end > len(filename):
                break
            end += 1

    return filename[start:end-1]

def _draw_network(mode='gnp', G=None, solution=None, n=0, edge_data=None):
    if mode == 'gnp':
        if not G:
            warnings.warn('Drawing gnp graphs requires G instance. Drawing skipped')
            return
    elif mode == 'reg':
        if not G:
            warnings.warn('Drawing gnp graphs requires G instance. Drawing skipped')
            return
    elif mode == 'poly':
        if not n:
            warnings.warn('Drawing poly graphs requires n. Drawing skipped.')
            return
        G = nx.Graph()
        G.add_nodes_from(range(n))
        edges = [(i, (i+1)%n, 1.) for i in range(n)]
        G.add_weighted_edges_from(edges)
        if not solution:
            solution = [i%2 for i in range(n)]
    elif mode == 'custom':
        if not n or not edge_data:
            warnings.warn('Drawing custom graphs requires n and edge_data. Drawing skipped.')
            return
        G = nx.Graph()
        G.add_nodes_from(range(n))
        G.add_weighted_edges_from(edge_data)

    if solution:
        colors = ['r' if solution[i] == 0 else 'b' for i in G.nodes()]
    else:
        warnings.warn('Solution not received. Drawing graph without solution.')
        colors = ['r' for node in G.nodes()]

    pos = nx.spring_layout(G)
    nx.draw_networkx(G, node_color=colors, node_size=600, alpha=1., pos=pos)

def maxcut_brute(G):
    n = len(G.nodes())
    w = nx.adjacency_matrix(G).toarray()

    # brute-force maxcut
    best_cost_brute = 0
    for b in range(2**n):
        x = [int(t) for t in reversed(list(bin(b)[2:].zfill(n)))]
        cost = 0
        for i in range(n):
            for j in range(n):
                cost = cost + w[i, j]*x[i]*(1-x[j])
        if best_cost_brute < cost:
            best_cost_brute = cost
            xbest_brute = x

    return best_cost_brute, xbest_brute

def _gnp_params(node, prob, seed=None):
    if seed is None:
        seed = 123

    G = nx.fast_gnp_random_graph(node, prob, seed=seed)
    print('Gnp instance: node=%d, prob=%s' % (node, prob))

    n_edge = len(G.edges())
    shift = -n_edge/2
    cost, solution = maxcut_brute(G)

    _draw_network(mode='gnp', G=G, solution=solution)

    return list(G.edges), shift, cost

def _reg_params(degree, node, seed=None):
    if seed is None:
        seed = 123

    G = nx.random_regular_graph(degree, node, seed=seed)
    print(f'Regular instance: node={node}, deg={degree}')

    n_edge = len(G.edges())
    shift = -n_edge/2
    cost, solution = maxcut_brute(G)

    _draw_network(mode='reg', G=G, solution=solution)

    return list(G.edges), shift, cost

def extract_from_filename(filename, data):
    """
    Extracts primary data from file names and add their info to the
    `data` dictionary.
    """
    mode = get_info(filename, '_', mode='get-alpha')
    n = int(get_info(filename, 'n'))
    seed = int(get_info(filename, 'seed'))
    data['node'] = n
    data['seed'] = seed

    if mode == 'poly':
        data['n_edge'] = n
        data['shift'] = -n/2.0
        data['true_obj'] = float(n) if n % 2 == 0 else float(n-1)
        print('Polygon instance: node={}'.format(n))
        _draw_network(mode=mode, n=n)
    elif mode == 'gnp':
        pr = get_info(filename, 'gnp')
        data['prob'] = float(pr)/(10.0**(len(pr)-1))
        data['edges'], data['shift'], data['true_obj'] = _gnp_params(data['node'], data['prob'], data['seed'])
        data['n_edge'] = len(data['edges'])
    elif mode == 'reg':
        data['degree'] = int(get_info(filename, 'reg'))
        data['edges'], data['shift'], data['true_obj'] = _reg_params(data['degree'], data['node'], data['seed'])
        data['n_edge'] = len(data['edges'])
    else:
        # custom graph
        data['shift'] = -data['n_edge']/2.0
        print('Custom graph instance, please set n_edge and true_obj manually before running this script.')

    return data

def load_data_prototype(mode, G=None):
    data = {
        'node': 0,
        'edges': [],
        'n_edge': 0,
        'shift': 0.0,
        'true_obj': 0.0,
        'obj': {},
        'energies': {},
        'obj_norm': {},
        'alpha': {},
        'params': {},
    }

    if mode == 'reg':
        data['degree'] = 0
    elif mode == 'gnp':
        data['prob'] = 0.0

    if G is not None:
        data['node'] = len(G.nodes)
        data['edges'] = list(G.edges)
        data['n_edge'] = len(G.edges)
        data['shift'] = -data['n_edge']/2.0

        if mode == 'reg':
            data['degree'] = len(list(G.neighbors(0)))

    return data
