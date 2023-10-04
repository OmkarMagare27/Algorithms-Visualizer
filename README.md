# Algorithms-Visualizer
"Explore various algorithms visually with this Python application utilizing Tkinter. Select from a range of algorithms like sorting, searching, and graph algorithms, and observe their execution in real-time through graphical representations. A valuable tool for understanding algorithmic operations and behavior by visual inspection."

Detailed Description:
1. Libraries Used:
Tkinter:

It is the standard GUI toolkit for Python. Tkinter provides powerful GUI-based tools that can be used to create desktop applications.
In this code, Tkinter is used to create the main window, canvas (to draw and visualize algorithms), buttons, and dropdowns.
Random:

This module is used to generate random numbers.
In this code, it's used to create arrays with random integers, which are then sorted using different algorithms.
Time:

This module provides various time-related functions.
Here, sleep function is used to introduce a delay in the visualization so that each step can be visualized clearly.
Queue:

This module provides the Queue data structure.
Used in the code for the breadth-first search algorithm to keep track of nodes to visit next.
2. Main Components:
Class AlgorithmVisualizer:

This class initializes the GUI and contains methods for each algorithm visualization.
It includes a method to draw an array, draw_array, which visualizes the array at each step of an algorithm.
Different algorithm methods like insertion_sort, bubble_sort, binary_search, etc. are defined, which handle the visualization and execution of respective algorithms.
Tkinter Widgets:

Canvas: A canvas widget is used to draw the visualization of arrays and graphs.
OptionMenu: A dropdown menu allows the user to select which algorithm to visualize.
Button: A run button is used to execute the selected algorithm.
Algorithm Methods:

Each algorithm method performs the following:
Generates an array (or other data structures) with random numbers.
Visualizes the data step-by-step as the algorithm progresses.
Implements the logic of the respective algorithm.
Calls draw_array method or other drawing methods (not fully implemented) to visualize each step.
Utility Methods:

draw_array: Draws a bar chart-like visualization of an array.
run_algorithm: Executes the algorithm chosen in the OptionMenu.
3. Algorithms:
Sorting Algorithms:

insertion_sort, bubble_sort, and others sort an array and visualize the process.
Search Algorithms:

binary_search: Searches an element in an array.
bfs: Searches using breadth-first search in a graph.
Graph Algorithms:

dijkstras_algorithm, kruskals_algorithm, and others operate on graph structures.
Mathematical Algorithms:

euclids_algorithm: Finds the Greatest Common Divisor (GCD) of two numbers.
Other Algorithms:

The code also contains various other algorithms like quickselect, flood_fill_algorithm, floyds_cycle_detection_algorithm, etc., each having its unique application.
4. Visualization:
Visualization is done by deleting and redrawing on the canvas during each step of the algorithm.
Delays (time.sleep(1)) are added to make transitions visible.
Actual drawing methods for some algorithms (like graph algorithms) need to be implemented by the user.
5. Execution:
After selecting an algorithm from the dropdown, clicking the "Run" button triggers the visualization of the selected algorithm.
Note:
The code provides a strong foundation for visualizing algorithms but may require additional adjustments or implementations for specific visualizations.
Ensure to run the code in a local environment that supports GUI to visualize the algorithms appropriately.

------------------------------------------------------------------------------------------------------------------------------------------------------------------------



### 1. Binary Search Algorithm:
- **Type**: Search
- **Description**: Divides a sorted array into two halves and checks if the desired element is in the left or right half. This process repeats until the element is found or the array size becomes 0.
- **Time Complexity**: \(O(\log n)\)

### 2. Breadth-First Search (BFS) Algorithm:
- **Type**: Graph Search
- **Description**: Explores all of the neighbor nodes at the present depth before moving on to nodes at the next depth level.
- **Time Complexity**: \(O(|V| + |E|)\), where \(V\) is vertices and \(E\) is edges.

### 3. Depth-First Search (DFS) Algorithm:
- **Type**: Graph Search
- **Description**: Explores as far as possible along each branch before backtracking.
- **Time Complexity**: \(O(|V| + |E|)\)

### 4. Merge Sort Algorithm:
- **Type**: Sort
- **Description**: Divides an array into two halves, recursively sorts them, and then merges them.
- **Time Complexity**: \(O(n \log n)\)

### 5. Quicksort Algorithm:
- **Type**: Sort
- **Description**: Selects a 'pivot' element and partitions the other elements into two sub-arrays, according to those less than the pivot and those greater than the pivot.
- **Time Complexity**: \(O(n \log n)\) average case, \(O(n^2)\) worst case.

### 6. Kruskal’s Algorithm:
- **Type**: Graph (Minimum Spanning Tree)
- **Description**: Finds the minimum spanning tree for a connected weighted graph. It adds the smallest weight edge that doesn’t form a cycle.
- **Time Complexity**: \(O(E \log V)\)

### 7. Floyd Warshall Algorithm:
- **Type**: Graph (Shortest Path)
- **Description**: Finds the shortest path between all pairs of vertices in a weighted graph.
- **Time Complexity**: \(O(V^3)\)

### 8. Dijkstra’s Algorithm:
- **Type**: Graph (Shortest Path)
- **Description**: Finds the shortest path from a source vertex to all vertices in a weighted graph.
- **Time Complexity**: \(O(V^2)\) without priority queue, \(O((V+E) \log V)\) with priority queue.

### 9. Bellman Ford Algorithm:
- **Type**: Graph (Shortest Path)
- **Description**: Computes shortest paths from a single source vertex to all other vertices in a weighted graph.
- **Time Complexity**: \(O(V \cdot E)\)

### 10. Kadane’s Algorithm:
- **Type**: Array (Maximum Subarray)
- **Description**: Finds the contiguous subarray within a one-dimensional array of numbers that has the largest sum.
- **Time Complexity**: \(O(n)\)

### 11. Lee Algorithm:
- **Type**: Graph Search (Shortest Path in a Maze)
- **Description**: Finds the shortest path in a maze from a starting position to a target, moving only along specified paths (often right-angled).
- **Time Complexity**: \(O(V + E)\)

### 12. Flood Fill Algorithm:
- **Type**: Image Processing / Graph Traversal
- **Description**: Changes the color of all connected pixels of a selected color to a target color.
- **Time Complexity**: \(O(n \times m)\) for an \(n \times m\) image.

### 13. Floyd’s Cycle Detection Algorithm:
- **Type**: Linked List (Cycle Detection)
- **Description**: Determines whether a linked list has a cycle using two pointers, which move at different speeds.
- **Time Complexity**: \(O(n)\)

### 14. Union Find Algorithm:
- **Type**: Disjoint Set
- **Description**: Manages a partition of a set into disjoint, non-overlapping subsets.
- **Time Complexity**: \(O(\log^* n)\) per operation with path compression and union by rank.

### 15. Topological Sort Algorithm:
- **Type**: Graph (Sorting Vertices)
- **Description**: Linearly orders the vertices of a directed acyclic graph.
- **Time Complexity**: \(O(V + E)\)

### 16. KMP Algorithm (Knuth-Morris-Pratt):
- **Type**: String Search (Substring search)
- **Description**: Searches for occurrences of a substring within a main string.
- **Time Complexity**: \(O(N + M)\), where \(N\) and \(M\) are the lengths of the main string and substring, respectively.

### 17. Insertion Sort Algorithm:
- **Type**: Sort
- **Description**: Builds the final sorted array one item at a time.
- **Time Complexity**: \(O(n^2)\)

### 18. Selection Sort Algorithm:
- **Type**: Sort
- **Description**: Repeatedly selects the smallest (or largest) element and moves it to the beginning (or end).
- **Time Complexity**: \(O(n^2)\)

### 19. Counting Sort Algorithm:
- **Type**: Sort
- **Description**: Sorts integers in linear time by counting the occurrences of each input.
- **Time Complexity**: \(O(n + k)\), where \(k\) is the range of the input.

### 20. Heap Sort Algorithm:
- **Type**: Sort
- **Description**: Builds a heap from the input data and then sorts the array using the heap.
- **Time Complexity**: \(O(n \log n)\)

### 21. Kahn’s Topological Sort Algorithm:
- **Type**: Graph (Sorting Vertices)
- **Description**: A variant of the Topological Sort Algorithm that works by choosing vertices in the same order as the eventual topological sort.
- **Time Complexity**: \(O(V + E)\)

### 22. Huffman Coding Compression Algorithm:
- **Type**: Data Compression
- **Description**: A lossless data compression algorithm that assigns variable-length codes to input characters based on their frequencies.
- **Time Complexity**: \(O(n \log n)\)

### 23. Quickselect Algorithm:
- **Type**: Selection Algorithm
- **Description**: Finds the kth smallest (or largest) element in an unordered list.
- **Time Complexity**: \(O(n)\) average case, \(O(n^2)\) worst case.

### 24. Boyer–Moore Majority Vote Algorithm:
- **Type**: Array
- **Description**: Finds the majority element (appears more than n/2 times) in an array.
- **Time Complexity**: \(O(n)\)

### 25. Euclid’s Algorithm:
- **Type**: Number Theory (GCD)
- **Description**: Computes the greatest common divisor of two integers.
- **Time Complexity**: \(O(\log \min(a, b))\)
