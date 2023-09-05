#include "widereach.h"
#include "helper.h"
#include <math.h>

prio_queue_t *create_prio_queue(int mt) {
  prio_queue_t *q = CALLOC(1, prio_queue_t);
  q->k = 4;
  q->sz = pow(2, q->k) - 1;
  q->n = 0;
  q->heap = CALLOC(q->sz, subproblem_t *);
  if(mt) {
    q->mt = 1;
    pthread_mutex_init(&q->lock, NULL);
  } else {
    q->mt = 0;
  }
  return q;
}

void prio_enlarge(prio_queue_t *q) {
  q->k++;
  q->sz = pow(2, q->k) - 1;
  q->heap = realloc(q->heap, q->sz * sizeof(subproblem_t *));
}

size_t prio_parent(size_t i) {
  return (i-1)/2;
}

void prio_swap(prio_queue_t *q, int i, int j) {
  subproblem_t *temp = q->heap[i];
  q->heap[i] = q->heap[j];
  q->heap[j] = temp;
}

size_t prio_get_n(prio_queue_t *q) {
  if(q->mt) {
    pthread_join(q->thread, NULL);
  }
  return q->n;
}

typedef struct enq_args_t {
  prio_queue_t *q;
  subproblem_t *prob;
} enq_args_t;

void *prio_enqueue_mt(void *args) {
  //this should be refactored - repeats code of prio_enqueue
  enq_args_t *argsf = args;
  prio_queue_t *q = argsf->q;
  subproblem_t *prob = argsf->prob;
  pthread_mutex_lock(&q->lock);
  if(q->n == q->sz)
    prio_enlarge(q);
  q->heap[q->n++] = prob;
  size_t i = q->n - 1;
  while(i > 0 && q->heap[prio_parent(i)]->score < q->heap[i]->score) {
    prio_swap(q, i, prio_parent(i));
    i = prio_parent(i);
  }
  pthread_mutex_unlock(&q->lock);
  free(args);
  return NULL;
}

void prio_enqueue(prio_queue_t *q, subproblem_t *prob) {
  if(q->mt) {
    pthread_join(q->thread, NULL);
    //enq_args_t args = {.q = q, .prob = prob};
    enq_args_t *args = CALLOC(1, enq_args_t);
    args->q = q;
    args->prob = prob;
    pthread_create(&q->thread, NULL, prio_enqueue_mt, args);
    return;
  }
  if(q->n == q->sz)
    prio_enlarge(q);
  q->heap[q->n++] = prob;
  size_t i = q->n - 1;
  while(i > 0 && q->heap[prio_parent(i)]->score < q->heap[i]->score) {
    prio_swap(q, i, prio_parent(i));
    i = prio_parent(i);
  }
}

void sift_down(prio_queue_t *q, int k) {
  if(q->n == 0) return;
  if(q->n - 1 == 2 * k + 1) {
    if(q->heap[2*k + 1] > q->heap[k])
      prio_swap(q, k, 2*k+1);
    return;
  }
  if(2*k + 2 > q->n - 1) return;
  int li;
  if(q->heap[2*k+1] > q->heap[2*k+2])
    li = 2*k+1;
  else
    li = 2*k+2;

  if(q->heap[li] > q->heap[k])
    prio_swap(q, k, li);

  sift_down(q, li);
}

typedef struct sift_args_t {
  prio_queue_t *q;
  int k;
} sift_args_t;

void *sift_down_mt(void *args) {
  sift_args_t *argsf = args;
  prio_queue_t *q = argsf->q;
  int k = argsf->k;
  pthread_mutex_lock(&q->lock);
  sift_down(q, k);
  pthread_mutex_unlock(&q->lock);
  free(args);
  return NULL;
}

subproblem_t *prio_pop_max(prio_queue_t *q) {
  if(prio_get_n(q) == 0)
    return NULL;
  if(q->mt) pthread_mutex_lock(&q->lock);
  subproblem_t *max = q->heap[0];
  q->heap[0] = q->heap[q->n-1];
  q->n--;
  if(q->mt) {
    pthread_mutex_unlock(&q->lock);
    pthread_join(q->thread, NULL);
    sift_args_t *args = CALLOC(1, sift_args_t);
    args->q = q;
    args->k = 0;
    pthread_create(&q->thread, NULL, sift_down_mt, args);
  } else {
    sift_down(q, 0);
  }
  return max;
}
