import tkinter as tk
import random
import time
import queue
import heapq
import collections

class AlgorithmVisualizer(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.canvas = tk.Canvas(self, width=600, height=400, bg='white')
        self.canvas.pack()
        self.algorithms = {
            "Insertion Sort": self.insertion_sort,
            "Selection Sort": self.heapify,
            "Heap Sort ": self.selection_sort,
            "Counting Sort": self.counting_sort,
            "Kahn's Topological Sort": self.kahns_topological_sort,
            "Euclids Algorithm": self.euclids_algorithm,
            "Quickselect Algorithm": self.quickselect,
            "Huffman Coding": self.huffman_coding,
            "Kruskals Algorithm": self.kruskals_algorithm,
            "Dijkstras Algorithm": self.dijkstras_algorithm,
            "Bellman Ford Algorithm": self.bellman_ford_algorithm,
            "Union Find Algorithm": self.union_find_algorithm,
            "Floyd Warshall": self.floyd_warshall,
            "KMP Algorithm": self.kmp_algorithm,
            "Flood Fill Algorithm": self.flood_fill_algorithm,
            "Floyd's Cycle Detection Algorithm": self.floyds_cycle_detection_algorithm,
            "Bubble Sort": self.bubble_sort,
            "Topological Sort Algorithm": self.topological_sort_algorithm,
            "Binary Search": self.binary_search,
            "BFS": self.bfs,
            "DFS": self.dfs,
            "Kadane's Algorithm": self.kadanes_algorithm,
            "Boyer_Moore Majority Vote Algorithm": self.majority_vote,
            "Merge Sort": self.merge_sort,
            "Quicksort": self.quick_sort,

            # Add more algorithms here...
        }
        self.algorithm_selector = tk.StringVar(self)
        self.algorithm_selector.set("Insertion Sort") # default value
        self.dropdown = tk.OptionMenu(self, self.algorithm_selector, *self.algorithms.keys())
        self.dropdown.pack()
        self.run_button = tk.Button(self, text="Run", command=self.run_algorithm)
        self.run_button.pack()
        
    def run_algorithm(self):
        algorithm_name = self.algorithm_selector.get()
        self.algorithms[algorithm_name]()
        
    def draw_array(self, arr, *args, **kwargs):
        self.canvas.delete("all")
        c_width = 600
        c_height = 400
        bar_width = c_width // len(arr)
        for i, val in enumerate(arr):
            x0 = i * bar_width
            y0 = c_height - val * 10
            x1 = (i + 1) * bar_width
            y1 = c_height
            self.canvas.create_rectangle(x0, y0, x1, y1, fill="green", **kwargs)
            self.canvas.create_text(x0 + bar_width/2, y0, anchor=tk.S, text=str(val), **kwargs)
        self.update_idletasks()
        time.sleep(1)

    
    
    def quickselect(self, arr, k):
        if len(arr) == 1:
            return arr[0]

        pivot = random.choice(arr)
        
        lows = [el for el in arr if el < pivot]
        highs = [el for el in arr if el > pivot]
        pivots = [el for el in arr if el == pivot]
        
        if k < len(lows):
            return self.quickselect(lows, k)
        elif k < len(lows) + len(pivots):
            # We got lucky and guessed a pivot
            return pivots[0]
        else:
            return self.quickselect(highs, k - len(lows) - len(pivots))
        # Visualize the current state of array

        
    def majority_vote(self, nums):
        count = 0
        candidate = None
        
        for num in nums:
            if count == 0:
                candidate = num
            count += (1 if num == candidate else -1)
            # Visualize count and candidate change
        return candidate

    
    
    
    
    def huffman_coding(self, data):
        frequency = collections.Counter(data)
        priority_queue = [[weight, [symbol, ""]] for symbol, weight in frequency.items()]
        heapq.heapify(priority_queue)
        while len(priority_queue) > 1:
            lo = heapq.heappop(priority_queue)
            hi = heapq.heappop(priority_queue)
            for pair in lo[1:]:
                pair[1] = '0' + pair[1]
            for pair in hi[1:]:
                pair[1] = '1' + pair[1]
            heapq.heappush(priority_queue, [lo[0] + hi[0]] + lo[1:] + hi[1:])
            # Draw the current priority queue or tree
        huff_tree = heapq.heappop(priority_queue)[1:]
        # You might visualize the final tree or code
    
    
    def union_find_algorithm(self, parent, i):
        self.draw_set(parent)  # Visualize
        self.update_idletasks()
        time.sleep(1)
        
        if parent[i] == i:
            return i
        return self.union_find_algorithm(parent, parent[i])

    def union(self, parent, x, y):
        x_root = self.union_find_algorithm(parent, x)
        y_root = self.union_find_algorithm(parent, y)
        parent[x_root] = y_root
        
        self.draw_set(parent)  # Visualize
        self.update_idletasks()
        time.sleep(1)

    
    def kahns_topological_sort(self):
        graph = {
            # Represent graph as an adjacency list
            # Example: 0: [1, 2], 1: [3], 2: [3], 3: []
        }
        
        in_degree = [0] * len(graph)
        for i in graph:
            for j in graph[i]:
                in_degree[j] += 1
        
        queue = []
        for i, degree in enumerate(in_degree):
            if degree == 0:
                queue.append(i)
        
        while queue:
            vertex = queue.pop(0)
            self.draw_graph(graph, highlight=vertex)  # Implement drawing method
            for i in graph[vertex]:
                in_degree[i] -= 1
                if in_degree[i] == 0:
                    queue.append(i)

    
    
    def heapify(self, arr, n, i):
        largest = i
        l = 2 * i + 1
        r = 2 * i + 2
        
        if l < n and arr[i] < arr[l]:
            largest = l
        
        if r < n and arr[largest] < arr[r]:
            largest = r
        
        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]
            self.draw_array(arr, highlight=(i, arr[i]))
            self.heapify(arr, n, largest)

    def heap_sort(self):
        arr = [random.randint(1, 30) for _ in range(20)]
        n = len(arr)
        
        for i in range(n//2 - 1, -1, -1):
            self.heapify(arr, n, i)
        
        for i in range(n-1, 0, -1):
            arr[i], arr[0] = arr[0], arr[i]
            self.draw_array(arr, highlight=(i, arr[i]))
            self.heapify(arr, i, 0)

        
        
    def counting_sort(self):
        arr = [random.randint(1, 30) for _ in range(20)]
        max_val = max(arr)
        count = [0] * (max_val + 1)
        output = [0] * len(arr)
        
        for num in arr:
            count[num] += 1
            self.draw_array(count)
        
        for i in range(1, len(count)):
            count[i] += count[i-1]
            self.draw_array(count)
        
        for num in reversed(arr):
            output[count[num] - 1] = num
            count[num] -= 1
            self.draw_array(output)




    def selection_sort(self):
        arr = [random.randint(1, 30) for _ in range(20)]
        for i in range(len(arr)):
            min_idx = i
            for j in range(i+1, len(arr)):
                if arr[min_idx] > arr[j]:
                    min_idx = j
            
            arr[i], arr[min_idx] = arr[min_idx], arr[i]
            self.draw_array(arr, highlight=(i, arr[i]))

    
    
    def kmp_algorithm(self, pat, txt):
        M = len(pat)
        N = len(txt)
        lps = [0]*M
        j = 0  # Index for pat[]
        
        # Preprocess the pattern
        self.compute_lps_array(pat, M, lps)
        
        i = 0  # Index for txt[]
        while i < N:
            if pat[j] == txt[i]:
                i += 1
                j += 1
            if j == M:
                # Pattern found
                self.draw_text(txt, i-j, i-j+M)  # Implement visualization
                self.update_idletasks()
                time.sleep(1)
                j = lps[j-1]
            elif i < N and pat[j] != txt[i]:
                if j != 0:
                    j = lps[j-1]
                else:
                    i += 1

    def compute_lps_array(self, pat, M, lps):
        length = 0
        lps[0] = 0
        i = 1
        
        while i < M:
            if pat[i] == pat[length]:
                length += 1
                lps[i] = length
                i += 1
            else:
                if length != 0:
                    length = lps[length-1]
                else:
                    lps[i] = 0
                    i += 1
            # Visualize LPS array
            self.draw_array(lps)
            self.update_idletasks()
            time.sleep(1)

    
    
    
    def flood_fill_algorithm(self, x, y, new_color, old_color):
        # Considering a 2D matrix grid for visualization
        if x < 0 or y < 0 or x >= len(self.grid) or y >= len(self.grid[0]) or self.grid[x][y] != old_color:
            return
        self.grid[x][y] = new_color
        self.canvas.delete("all")
        self.draw_grid()  # Implement draw_grid() to visualize the grid
        self.update_idletasks()
        time.sleep(1)
        
        self.flood_fill_algorithm(x+1, y, new_color, old_color)
        self.flood_fill_algorithm(x-1, y, new_color, old_color)
        self.flood_fill_algorithm(x, y+1, new_color, old_color)
        self.flood_fill_algorithm(x, y-1, new_color, old_color)

    
    
    def floyds_cycle_detection_algorithm(self, node):
        slow_pointer = node
        fast_pointer = node
        while fast_pointer and fast_pointer.next_node:
            slow_pointer = slow_pointer.next_node
            fast_pointer = fast_pointer.next_node.next_node
            if slow_pointer == fast_pointer:
                # Cycle detected
                self.draw_linked_list(node, cycle=True)  # Implement visualization
                self.update_idletasks()
                time.sleep(1)
                return True
        # No cycle detected
        self.draw_linked_list(node, cycle=False)  # Implement visualization
        self.update_idletasks()
        time.sleep(1)
        return False

    
    
    def bellman_ford_algorithm(self):
        graph = {0: [(1, -1), (2, 4)],
                1: [(2, 3), (3, 2), (4, 2)],
                2: [],
                3: [(2, 5), (1, 1)],
                4: [(3, -3)]}
        
        source_vertex = 0
        distance_to_vertices = [float("inf")] * len(graph)
        distance_to_vertices[source_vertex] = 0

        for _ in range(len(graph) - 1):
            for vertex in graph:
                for neighbor, cost in graph[vertex]:
                    if distance_to_vertices[vertex] + cost < distance_to_vertices[neighbor]:
                        distance_to_vertices[neighbor] = distance_to_vertices[vertex] + cost
            
            self.canvas.delete("all")
            self.draw_graph_bellman_ford(graph, distance_to_vertices)
            self.update_idletasks()
            time.sleep(1)

    
    
    def dijkstras_algorithm(self):
        # Sample graph: (node1, node2, weight)
        graph = {0: [(1, 1), (2, 4)],
                1: [(2, 2), (3, 5)],
                2: [(3, 3)],
                3: []}
        
        start_vertex = 0
        shortest_paths = {start_vertex: (None, 0)}
        current_vertex = start_vertex
        visited_vertices = set()
        
        while current_vertex is not None:
            visited_vertices.add(current_vertex)
            neighbors = graph[current_vertex]
            current_weight = shortest_paths[current_vertex][1]

            for neighbor, weight in neighbors:
                if neighbor not in visited_vertices:
                    new_weight = current_weight + weight
                    if neighbor not in shortest_paths:
                        shortest_paths[neighbor] = (current_vertex, new_weight)
                    else:
                        current_shortest_weight = shortest_paths[neighbor][1]
                        if current_shortest_weight > new_weight:
                            shortest_paths[neighbor] = (current_vertex, new_weight)
            
            self.canvas.delete("all")
            self.draw_graph_dijkstra(graph, visited_vertices, shortest_paths)
            self.update_idletasks()
            time.sleep(1)
            
            next_vertices = {vertex: weight for vertex, weight in shortest_paths.items() if vertex not in visited_vertices}
            if not next_vertices:
                current_vertex = None
            else:
                current_vertex = min(next_vertices, key=lambda k: next_vertices[k][1])

    
    def floyd_warshall(self):
        # Sample graph in adjacency matrix format
        graph = [
            [0, 1, float('inf'), 1],
            [1, 0, 1, float('inf')],
            [float('inf'), 1, 0, 1],
            [1, float('inf'), 1, 0]
        ]
        num_vertices = len(graph)
        
        for k in range(num_vertices):
            for i in range(num_vertices):
                for j in range(num_vertices):
                    graph[i][j] = min(graph[i][j], graph[i][k] + graph[k][j])
                    self.canvas.delete("all")
                    self.draw_graph_fw(graph, i, j, k)
                    self.update_idletasks()
                    time.sleep(0.5)

    
    
    def topological_sort_algorithm(self, graph):
        stack = []
        visited = [False] * len(graph)
        
        for i in range(len(graph)):
            if visited[i] == False:
                self.topological_sort_util(i, visited, stack, graph)
        
        # Visualizing the stack (the topological order)
        self.draw_stack(stack)
        self.update_idletasks()
        time.sleep(1)

    def topological_sort_util(self, v, visited, stack, graph):
        visited[v] = True
        
        for i in graph[v]:
            if visited[i] == False:
                self.topological_sort_util(i, visited, stack, graph)
        
        stack.insert(0, v)
        # Visualizing the stack
        self.draw_stack(stack)
        self.update_idletasks()
        time.sleep(1)

    
    
    def quick_sort(self, arr=None, low=None, high=None):
        if arr is None:
            arr = [random.randint(1, 100) for _ in range(12)]
            low = 0
            high = len(arr) - 1
        
        if low < high:
            pi = self.partition(arr, low, high)
            self.quick_sort(arr, low, pi-1)
            self.quick_sort(arr, pi+1, high)

    def partition(self, arr, low, high):
        self.canvas.delete("all")
        self.draw_array(arr)
        self.update_idletasks()
        time.sleep(0.5)
        
        pivot = arr[high]
        i = (low - 1)
        for j in range(low, high):
            if arr[j] <= pivot:
                i = i + 1
                arr[i], arr[j] = arr[j], arr[i]
        arr[i + 1], arr[high] = arr[high], arr[i + 1]
        return (i + 1)


    def merge_sort(self, arr=None):
        if arr is None:
            arr = [random.randint(1, 100) for _ in range(12)]
        if len(arr) > 1:
            mid = len(arr) // 2
            L = arr[:mid]
            R = arr[mid:]
            self.merge_sort(L)
            self.merge_sort(R)
            
            i = j = k = 0
            
            while i < len(L) and j < len(R):
                self.canvas.delete("all")
                self.draw_array(arr)
                self.update_idletasks()
                time.sleep(0.5)
                
                if L[i] < R[j]:
                    arr[k] = L[i]
                    i += 1
                else:
                    arr[k] = R[j]
                    j += 1
                k += 1
            
            while i < len(L):
                self.canvas.delete("all")
                self.draw_array(arr)
                self.update_idletasks()
                time.sleep(0.5)
                
                arr[k] = L[i]
                i += 1
                k += 1
            
            while j < len(R):
                self.canvas.delete("all")
                self.draw_array(arr)
                self.update_idletasks()
                time.sleep(0.5)
                
                arr[k] = R[j]
                j += 1
                k += 1

        
    def insertion_sort(self):
        arr = [random.randint(1, 30) for _ in range(20)]
        for i in range(1, len(arr)):
            key = arr[i]
            j = i-1
            while j >= 0 and key < arr[j]:
                self.draw_array(arr)
                arr[j + 1] = arr[j]
                j -= 1
            arr[j + 1] = key
    
    def euclids_algorithm(self):
        a, b = 1980, 1617  # Example values
        while b:
            self.canvas.delete("all")
            self.canvas.create_text(300, 190, text=f'a: {a}, b: {b}', font=("Arial", 16))
            self.update_idletasks()
            time.sleep(1)
            a, b = b, a % b
        self.canvas.create_text(300, 210, text=f'gcd: {a}', font=("Arial", 16))
    
    def bubble_sort(self):
        arr = [random.randint(10, 100) for _ in range(12)]
        for _ in range(len(arr)-1):
            for j in range(len(arr)-1):
                self.canvas.delete("all")
                self.draw_array(arr)
                self.update_idletasks()
                time.sleep(0.5)
                if arr[j] > arr[j+1]:
                    arr[j], arr[j+1] = arr[j+1], arr[j]

    def binary_search(self):
        arr = [i for i in range(12)]
        target = 6  # Example target
        left, right = 0, len(arr)-1
        while left <= right:
            self.canvas.delete("all")
            self.draw_array(arr, left, right)
            self.update_idletasks()
            time.sleep(1)
            mid = (left + right) // 2
            if arr[mid] == target:
                break
            elif arr[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
    
    def dfs(self, graph=None, node=None, visited=None):
        if graph is None:
            graph = {
                0: [1, 2],
                1: [0, 3, 4],
                2: [0, 4],
                3: [1, 5],
                4: [1, 2],
                5: [3]
            }
            node = 0
            visited = [False] * len(graph)
        
        if not visited[node]:
            self.canvas.delete("all")
            self.draw_graph(graph, visited)
            self.update_idletasks()
            time.sleep(1)
            visited[node] = True
            
            for neighbor in graph[node]:
                self.dfs(graph, neighbor, visited)

    
    def bfs(self):
        graph = {
            0: [1, 2],
            1: [0, 3, 4],
            2: [0, 4],
            3: [1, 5],
            4: [1, 2],
            5: [3]
        }
        start_node = 0
        visited = [False] * len(graph)
        q = queue.Queue()
        q.put(start_node)
        visited[start_node] = True
        
        while not q.empty():
            node = q.get()
            self.canvas.delete("all")
            self.draw_graph(graph, visited)
            self.update_idletasks()
            time.sleep(1)
            for neighbor in graph[node]:
                if not visited[neighbor]:
                    q.put(neighbor)
                    visited[neighbor] = True
    
    def draw_graph(self, graph, visited):
        self.canvas.delete("all")
        nodes = [(random.randint(50, 550), random.randint(50, 350)) for _ in range(len(graph))]
        for i, (x, y) in enumerate(nodes):
            self.canvas.create_oval(x-20, y-20, x+20, y+20, fill="green" if visited[i] else "red")
            self.canvas.create_text(x, y, text=str(i), fill="white")
        for node, neighbors in graph.items():
            x0, y0 = nodes[node]
            for neighbor in neighbors:
                x1, y1 = nodes[neighbor]
                self.canvas.create_line(x0, y0, x1, y1)
    
    def kruskals_algorithm(self):
        # Sample graph: (node1, node2, weight)
        edges = [(0, 1, 2), (0, 2, 3), (1, 3, 1), (1, 2, 4), (2, 3, 5)]
        edges.sort(key=lambda x: x[2])
        
        parent = list(range(4))  # Example with 4 nodes
        
        def find_parent(node):
            if parent[node] == node:
                return node
            return find_parent(parent[node])
        
        def union(node1, node2):
            root1 = find_parent(node1)
            root2 = find_parent(node2)
            if root1 != root2:
                parent[root1] = root2
        
        mst = []
        for edge in edges:
            node1, node2, weight = edge
            if find_parent(node1) != find_parent(node2):
                union(node1, node2)
                mst.append(edge)
                self.canvas.delete("all")
                self.draw_graph_mst(edges, mst)
                self.update_idletasks()
                time.sleep(1)

    
    def kadanes_algorithm(self):
        arr = [random.randint(-10, 10) for _ in range(10)]
        max_current = max_global = arr[0]
        start = end = s = 0
        for i in range(1, len(arr)):
            self.canvas.delete("all")
            self.draw_array(arr, s, end)
            self.update_idletasks()
            time.sleep(1)
            if arr[i] > (max_current + arr[i]):
                max_current = arr[i]
                s = i
            else:
                max_current = max_current + arr[i]
            if max_current > max_global:
                start = s
                end = i
                max_global = max_current
        
# Example usage:
app = AlgorithmVisualizer()
app.mainloop()
