package antarys

import (
	"math"
	"sort"
)

type queueItem struct {
	nodeID     uint32
	similarity float32
}

type priorityQueue struct {
	items []queueItem
	size  int
}

func (pq *priorityQueue) len() int {
	return pq.size
}

func (pq *priorityQueue) push(x queueItem) {

	if math.IsNaN(float64(x.similarity)) {
		return
	}

	if pq.size >= len(pq.items) {
		pq.items = append(pq.items, x)
	} else {
		pq.items[pq.size] = x
	}

	pq.size++
	pq.siftUpValidated(pq.size - 1)
}

func (pq *priorityQueue) siftUpValidated(i int) {
	if i == 0 {
		return
	}

	item := pq.items[i]

	for i > 0 {
		parent := (i - 1) / 2

		if parent < 0 || parent >= pq.size {
			break
		}

		if item.similarity <= pq.items[parent].similarity {
			break
		}

		pq.items[i] = pq.items[parent]
		i = parent
	}

	pq.items[i] = item
}

func (pq *priorityQueue) pop() queueItem {
	if pq.size == 0 {
		return queueItem{}
	}

	result := pq.items[0]

	pq.size--
	if pq.size > 0 {
		pq.items[0] = pq.items[pq.size]
		pq.siftDownFast(0)
	}

	return result
}

func (pq *priorityQueue) top() queueItem {
	if pq.size == 0 {
		return queueItem{}
	}
	return pq.items[0]
}

func (pq *priorityQueue) bottom() queueItem {
	if pq.size == 0 {
		return queueItem{similarity: -1.0}
	}

	if pq.size == 1 {
		return pq.items[0]
	}

	minIdx := 0
	minSim := pq.items[0].similarity

	startLeaf := pq.size / 2
	for i := startLeaf; i < pq.size; i++ {
		if pq.items[i].similarity < minSim {
			minSim = pq.items[i].similarity
			minIdx = i
		}
	}

	return pq.items[minIdx]
}

func (pq *priorityQueue) popBottom() {
	if pq.size == 0 {
		return
	}

	if pq.size == 1 {
		pq.size = 0
		return
	}

	lowestIdx := 0
	lowestSim := pq.items[0].similarity

	i := 1
	for ; i+3 < pq.size; i += 4 {
		if pq.items[i].similarity < lowestSim {
			lowestIdx = i
			lowestSim = pq.items[i].similarity
		}
		if pq.items[i+1].similarity < lowestSim {
			lowestIdx = i + 1
			lowestSim = pq.items[i+1].similarity
		}
		if pq.items[i+2].similarity < lowestSim {
			lowestIdx = i + 2
			lowestSim = pq.items[i+2].similarity
		}
		if pq.items[i+3].similarity < lowestSim {
			lowestIdx = i + 3
			lowestSim = pq.items[i+3].similarity
		}
	}

	for ; i < pq.size; i++ {
		if pq.items[i].similarity < lowestSim {
			lowestIdx = i
			lowestSim = pq.items[i].similarity
		}
	}

	pq.size--
	if lowestIdx != pq.size {
		pq.items[lowestIdx] = pq.items[pq.size]

		if lowestIdx < pq.size/2 {

			if !pq.siftDownFast(lowestIdx) {

				pq.siftUpFast(lowestIdx)
			}
		} else {

			pq.siftUpFast(lowestIdx)
		}
	}
}

func (pq *priorityQueue) siftUpFast(i int) {
	if i == 0 {
		return
	}

	item := pq.items[i]

	for i > 0 {
		parent := (i - 1) / 2

		if item.similarity <= pq.items[parent].similarity {
			break
		}

		pq.items[i] = pq.items[parent]
		i = parent
	}

	pq.items[i] = item
}

func (pq *priorityQueue) siftDownFast(i int) bool {
	if i >= pq.size/2 {
		return false
	}

	item := pq.items[i]
	half := pq.size / 2
	moved := false

	for i < half {
		left := 2*i + 1
		right := left + 1
		largest := i

		if left < pq.size && pq.items[left].similarity > pq.items[largest].similarity {
			largest = left
		}

		if right < pq.size && pq.items[right].similarity > pq.items[largest].similarity {
			largest = right
		}

		if largest == i {
			break
		}

		pq.items[i] = pq.items[largest]
		i = largest
		moved = true
	}

	pq.items[i] = item
	return moved
}

func (pq *priorityQueue) pushBulk(items []queueItem) {
	needed := pq.size + len(items)
	if needed > len(pq.items) {
		newItems := make([]queueItem, pq.size, needed)
		copy(newItems, pq.items[:pq.size])
		pq.items = newItems
	}

	for _, item := range items {
		pq.items[pq.size] = item
		pq.size++
	}

	for i := (pq.size - 2) / 2; i >= 0; i-- {
		pq.siftDownFast(i)
	}
}

func (pq *priorityQueue) clear() {
	pq.size = 0

}

func (pq *priorityQueue) peek(index int) queueItem {
	if index >= 0 && index < pq.size {
		return pq.items[index]
	}
	return queueItem{}
}

func (pq *priorityQueue) cap() int {
	return cap(pq.items)
}

func (pq *priorityQueue) toSortedSlice() []queueItem {
	if pq.size == 0 {
		return nil
	}

	result := make([]queueItem, pq.size)

	tempPQ := &priorityQueue{
		items: make([]queueItem, pq.size),
		size:  pq.size,
	}
	copy(tempPQ.items, pq.items[:pq.size])

	for i := 0; i < len(result); i++ {
		result[i] = tempPQ.pop()
	}

	return result
}

func (pq *priorityQueue) removeBottomN(n int) {
	if n <= 0 || n >= pq.size {
		if n >= pq.size {
			pq.clear()
		}
		return
	}

	if n <= 3 {
		for i := 0; i < n; i++ {
			pq.popBottom()
		}
		return
	}

	indices := make([]int, 0, n)
	similarities := make([]float32, pq.size)

	for i := 0; i < pq.size; i++ {
		similarities[i] = pq.items[i].similarity
	}

	for round := 0; round < n; round++ {
		minIdx := -1
		minSim := float32(math.MaxFloat32)

		for i := 0; i < pq.size; i++ {
			skip := false
			for _, selected := range indices {
				if i == selected {
					skip = true
					break
				}
			}
			if skip {
				continue
			}

			if similarities[i] < minSim {
				minSim = similarities[i]
				minIdx = i
			}
		}

		if minIdx >= 0 {
			indices = append(indices, minIdx)
		}
	}

	sort.Slice(indices, func(i, j int) bool {
		return indices[i] > indices[j]
	})

	for _, idx := range indices {
		pq.size--
		if idx != pq.size {
			pq.items[idx] = pq.items[pq.size]

			if idx < pq.size/2 {
				if !pq.siftDownFast(idx) {
					pq.siftUpFast(idx)
				}
			} else {
				pq.siftUpFast(idx)
			}
		}
	}
}
