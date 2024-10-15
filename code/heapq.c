#include "widereach.h"
#include "helper.h"
#include <math.h>
#include <assert.h>

//this is very similar to priority_queue.c - can refactor

heapq_t *create_heapq(int max_sz) {
  //max_sz = -1 => uncapped
  heapq_t *q = CALLOC(1, heapq_t);
  q->k = 4;
  q->sz = pow(2, q->k) - 1;
  q->n = 0;
  q->heap = CALLOC(q->sz, tnode *);
  q->max_sz = max_sz;
  return q;
}

void heapq_enlarge(heapq_t *q) {
  q->k++;
  q->sz = pow(2, q->k) - 1;
  q->heap = realloc(q->heap, q->sz * sizeof(tnode *));
  if(q->sz >= q->max_sz)
    printf("reached size cap\n");
}

size_t heapq_parent(size_t i) {
  return (i-1)/2;
}

void heapq_swap(heapq_t *q, int i, int j) {
  tnode *temp = q->heap[i];
  q->heap[i] = q->heap[j];
  q->heap[j] = temp;
}

size_t heapq_get_n(heapq_t *q) {
  return q->n;
}

void heapq_enqueue(heapq_t *q, tnode *node) {
  node->in_q = 1;
  if(q->n == q->sz)
    heapq_enlarge(q);
  q->heap[q->n++] = node;
  size_t i = q->n - 1;
  while(i > 0 && q->heap[heapq_parent(i)]->score < q->heap[i]->score) {
    heapq_swap(q, i, heapq_parent(i));
    i = heapq_parent(i);
  }
  if(q->max_sz > 0 && q->n >= q->max_sz) {
    tree_remove_node(q->heap[q->n-1]);
    free(free_subtree(q->heap[q->n-1]));
    q->n--; //when we hit the cap, remove the worst element
  }
}

void heapq_sift_down(heapq_t *q, int k) {
  if(q->n == 0) return;
  if(q->n - 1 == 2 * k + 1) {
    if(q->heap[2*k + 1]->score > q->heap[k]->score)
      heapq_swap(q, k, 2*k+1);
    return;
  }
  if(2*k + 2 > q->n - 1) return;
  int li;
  if(q->heap[2*k+1]->score > q->heap[2*k+2]->score)
    li = 2*k+1;
  else
    li = 2*k+2;

  if(q->heap[li]->score > q->heap[k]->score)
    heapq_swap(q, k, li);

  heapq_sift_down(q, li);
}

tnode *heapq_peak_bottom(heapq_t *q) {
  if(q->n == 0) return NULL;
  return q->heap[q->n-1];
}

double heapq_max_score(heapq_t *q) {
  double max = -1e101;
  for(int i = 0; i < q->n; i++) {
    if(q->heap[i]->score > max) {
      max = q->heap[i]->score;
    }
  }
  return max;
}

tnode *heapq_pop_max(heapq_t *q) {
  if(heapq_get_n(q) == 0)
    return NULL;
  tnode *max = q->heap[0];
  q->heap[0] = q->heap[q->n-1];
  q->n--;
  heapq_sift_down(q, 0);
  max->in_q = 0;
  return max;
}

int heapq_sift_up(heapq_t *q, int k) {
  //moves the item at idx up if it is greater than its parent, and repeats until it no longer is
  if(k == 0) return 0;
  int pi = (k-1)/2; //parent index
  if(q->heap[pi]->score < q->heap[k]->score) {
    heapq_swap(q, k, pi);
    return heapq_sift_up(q, pi);
  } else {
    return k;
  }
}

tnode *heapq_pop_epsilon(heapq_t *q, double epsilon) {
  //pops max, or with probability epsilon, pops a random element
  //to pop a random element, we replace it with the last one in the array, then sift up until the current element is smaller than its parent, then sift down from there
  if(heapq_get_n(q) == 0)
    return NULL;
  if(((double) rand())/RAND_MAX > epsilon)
    return heapq_pop_max(q);

  int idx = rand() % q->n;
  tnode *out = q->heap[idx];
  q->heap[idx] = q->heap[q->n-1];
  q->n--;
  int new_idx = heapq_sift_up(q, idx);
  heapq_sift_down(q, new_idx);
  out->in_q = 0;
  return out;
}

heapq_t *heapq_free(heapq_t *q) {
  //TODO: free the nodes
  free(q->heap);
  return q;
}
