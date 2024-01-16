#include "widereach.h"
#include "helper.h"

#include <memory.h>
#include <gsl/gsl_blas.h>
#include <math.h>

/* This file implements a union find data structure. It uses a slow list-based implementation, which is okay because it will only be run once */
typedef struct linked_list_t linked_list_t;
typedef struct llnode llnode;

typedef struct llnode {
  gsl_vector *pt;
  llnode *next;
  linked_list_t *owner;
  int class, i;
} llnode;

typedef struct linked_list_t {
  llnode *head;
  llnode *tail;
  int idx, n, old_idx, npos, nneg;
  double *dists;
} linked_list_t;

typedef struct union_find_t {
  int max_sets;
  int n_sets;
  linked_list_t **sets;
} union_find_t;

union_find_t *create_union_find(int max_sets) {
  union_find_t *uf = CALLOC(1, union_find_t);
  uf->n_sets = 0;
  uf->max_sets = max_sets;
  uf->sets = CALLOC(max_sets, linked_list_t *);
  return uf;
}

void make_singleton_set(union_find_t *uf, vsamples_t *vs, gsl_vector *x, int class, int i) {
  if(uf->n_sets >= uf->max_sets) {
    printf("Attempted to overload union find\n");
    return;
  }
  uf->sets[uf->n_sets] = CALLOC(1, linked_list_t);
  linked_list_t *ll = uf->sets[uf->n_sets];
  llnode *node = CALLOC(1, llnode);
  node->pt = gsl_vector_alloc(x->size);
  gsl_vector_memcpy(node->pt, x);
  node->class = class;
  node->i = i;
  node->owner = ll;
  ll->head = node;
  ll->tail = node;
  ll->idx = uf->n_sets;
  ll->n = 1;
  ll->dists = CALLOC(uf->max_sets, double);
  if(vs->label[class] == 1) {
    ll->npos = 1;
    ll->nneg = 0;
  } else {
    ll->nneg = 1;
    ll->npos = 0;
  }
  uf->n_sets++;
}

void make_sets_from_samples(union_find_t *uf, vsamples_t *vs) {
  for(int class = 0; class < vs->class_cnt; class++) {
    for(int i = 0; i < vs->count[class]; i++) {
      make_singleton_set(uf, vs, vs->samples[class][i], class, i);
    }
  }
}

void uf_populate_dists(union_find_t *uf) {
  //this should be called only when uf has singleton sets
  for(int i = 0; i < uf->n_sets; i++) {
    for(int j = i + 1; j < uf->n_sets; j++) {
      gsl_vector *u = uf->sets[i]->head->pt, *v = uf->sets[j]->head->pt;
      gsl_vector *diff = gsl_vector_alloc(u->size);
      gsl_vector_memcpy(diff, v);
      gsl_blas_daxpy(-1, u, diff);
      double dist = gsl_blas_dnrm2(diff);
      uf->sets[i]->dists[j] = dist;
      uf->sets[j]->dists[i] = dist;
      gsl_vector_free(diff);
    }
  }
}

void uf_populate_dists_sep_classes(union_find_t *uf) {
  //this should be called only when uf has singleton sets
  for(int i = 0; i < uf->n_sets; i++) {
    for(int j = i + 1; j < uf->n_sets; j++) {
      if(uf->sets[i]->head->class != uf->sets[j]->head->class) {
	uf->sets[i]->dists[j] = 1e101;
	uf->sets[j]->dists[i] = 1e101;
	continue;
      }
      gsl_vector *u = uf->sets[i]->head->pt, *v = uf->sets[j]->head->pt;
      gsl_vector *diff = gsl_vector_alloc(u->size);
      gsl_vector_memcpy(diff, v);
      gsl_blas_daxpy(-1, u, diff);
      double dist = gsl_blas_dnrm2(diff);
      uf->sets[i]->dists[j] = dist;
      uf->sets[j]->dists[i] = dist;
      gsl_vector_free(diff);
    }
  }
}

linked_list_t *uf_find(union_find_t *uf, llnode *x) {
  return x->owner;
}

void ll_append_node(linked_list_t *u, llnode *x) {
  llnode *tail = u->tail;
  tail->next = x;
  u->tail = x;
  x->owner = u;
  x->next = NULL;
  u->n++;
}

void ll_merge_into(linked_list_t *u, linked_list_t *v) {
  llnode *trav = v->head;
  llnode *next = NULL;
  while(trav) {
    next = trav->next;
    ll_append_node(u, trav);
    trav = next;
  }
  u->npos += v->npos;
  u->nneg += v->nneg;
}

void uf_union(union_find_t *uf, linked_list_t *u, linked_list_t *v) {
  if(u->idx == v->idx) {
    printf("Same index in union\n");
    return;
  }
  int removed_idx = -1, keep_idx = -1;
  double inter_dist = u->dists[v->idx];
  if(u->n > v->n) {
    ll_merge_into(u, v);
    removed_idx = v->idx;
    keep_idx = u->idx;
  } else {
    ll_merge_into(v, u);
    removed_idx = u->idx;
    keep_idx = u->idx;
  }

  for(int i = 0; i < uf->n_sets; i++) {
    if(i == removed_idx || i == keep_idx) continue;
    double old_dist1 = uf->sets[i]->dists[keep_idx];
    double old_dist2 = uf->sets[i]->dists[removed_idx];
    //Lance-Williams flexible method with a1 = a2 = 1/2, b = 1/4, per paper
    //double new_dist = 0.5*old_dist1 + 0.5*old_dist2 + 0*inter_dist - 0.5*fabs(old_dist1 - old_dist2);
    double new_dist = 0.5*old_dist1 + 0.5*old_dist2 + 0.25*inter_dist - 0*fabs(old_dist1 - old_dist2);
    uf->sets[i]->dists[keep_idx] = new_dist;
    uf->sets[keep_idx]->dists[i] = new_dist;

    //remove cluster from all the distance lists
    for(int j = removed_idx; j < uf->n_sets; j++) {
      if(j == uf->max_sets - 1)
	uf->sets[i]->dists[j] = 0;
      else
	uf->sets[i]->dists[j] = uf->sets[i]->dists[j+1];
    }
  }
  //remove cluster from keep's distance list
  for(int j = removed_idx; j < uf->n_sets; j++) {
    if(j == uf->max_sets - 1)
      uf->sets[keep_idx]->dists[j] = 0;
    else
      uf->sets[keep_idx]->dists[j] = uf->sets[keep_idx]->dists[j+1];
  }

  //remove cluster from the list of sets
  free(uf->sets[removed_idx]);
  for(int i = removed_idx; i < uf->n_sets; i++) {
    if(i == uf->max_sets - 1)
      uf->sets[i] = NULL;
    else
      uf->sets[i] = uf->sets[i+1];
    if(uf->sets[i])
      uf->sets[i]->idx--;
  }
  uf->n_sets--;
}

rounded_pts_t cluster_rounded_points(env_t *env, vsamples_t *vs, size_t n) {
  size_t d = env->samples->dimension;
  if(n > samples_total(env->samples)) {
    printf("ERR: too many round pts\n");
    return (rounded_pts_t) {0};
  }
  union_find_t *uf = create_union_find(samples_total(env->samples));
  make_sets_from_samples(uf, vs);
  uf_populate_dists(uf);
  
  while(uf->n_sets > n) {
    //find closest pair
    double min_dist = 1e101;
    int min_i, min_j;
    for(int i = 0; i < uf->n_sets; i++) {
      for(int j = i + 1; j < uf->n_sets; j++) {
	double dist = uf->sets[i]->dists[j];
	if(dist < min_dist) {
	  min_dist = dist;
	  min_i = i;
	  min_j = j;
	}
      }
    }
    uf_union(uf, uf->sets[min_i], uf->sets[min_j]);
  }
  
  round_pts_t round_pts;
  round_pts.n = n;
  gsl_vector **pts = CALLOC(n, gsl_vector *);
  round_pts.pts = pts;
  vsamples_t *rvs = malloc(sizeof(vsamples_t));
  rvs->dimension = d;
  rvs->class_cnt = vs->class_cnt;
  rvs->count = CALLOC(vs->class_cnt, size_t);
  memcpy(rvs->count, vs->count, vs->class_cnt*sizeof(size_t));
  rvs->label = CALLOC(vs->class_cnt, int);
  memcpy(rvs->label, vs->label, vs->class_cnt*sizeof(int));
  rvs->samples = CALLOC(vs->class_cnt, gsl_vector **);
  for(int class = 0; class < vs->class_cnt; class++) {
    rvs->samples[class] = CALLOC(vs->count[class], gsl_vector *);
  }

  round_pts.npos = CALLOC(n, int);
  round_pts.nneg = CALLOC(n, int);

  for(int i = 0; i < uf->n_sets; i++) {
    gsl_vector *avg = gsl_vector_calloc(vs->dimension);
    llnode *trav = uf->sets[i]->head;
    int npos = 0, nneg = 0;
    while(trav) {
      gsl_blas_daxpy(1, trav->pt, avg);
      //TODO: replace this with the npos and nneg struct fields already in the sets
      if(vs->label[trav->class] == 1)
	npos++;
      else
	nneg++;
      trav = trav->next;
    }
    gsl_vector_scale(avg, 1./uf->sets[i]->n);
    trav = uf->sets[i]->head;
    while(trav) {
      rvs->samples[trav->class][trav->i] = gsl_vector_calloc(vs->dimension);
      gsl_vector_memcpy(rvs->samples[trav->class][trav->i], avg);
      trav = trav->next;
    }
    pts[i] = avg;
    round_pts.npos[i] = npos;
    round_pts.nneg[i] = nneg;
  }
  
  return (rounded_pts_t) {round_pts, rvs};
}

rounded_pts_t cluster_rounded_points_consistent(env_t *env, vsamples_t *vs) {
  //creates clusters until we would combine a negative point with a positive one
  size_t d = env->samples->dimension;
  union_find_t *uf = create_union_find(samples_total(env->samples));
  make_sets_from_samples(uf, vs);
  uf_populate_dists(uf);

  int n = samples_total(env->samples);
  
  while(1) {
    //find closest pair
    double min_dist = 1e101;
    int min_i, min_j;
    for(int i = 0; i < uf->n_sets; i++) {
      for(int j = i + 1; j < uf->n_sets; j++) {
	double dist = uf->sets[i]->dists[j];
	if(dist < min_dist) {
	  min_dist = dist;
	  min_i = i;
	  min_j = j;
	}
      }
    }

    if((uf->sets[min_i]->npos > 0 && uf->sets[min_j]->nneg > 0) || (uf->sets[min_i]->nneg > 0 && uf->sets[min_j] < 0))
      break; //break if we would combine positive and negative points

    uf_union(uf, uf->sets[min_i], uf->sets[min_j]);
    n--;
  }
  
  round_pts_t round_pts;
  round_pts.n = n;
  gsl_vector **pts = CALLOC(n, gsl_vector *);
  round_pts.pts = pts;
  vsamples_t *rvs = malloc(sizeof(vsamples_t));
  rvs->dimension = d;
  rvs->class_cnt = vs->class_cnt;
  rvs->count = CALLOC(vs->class_cnt, size_t);
  memcpy(rvs->count, vs->count, vs->class_cnt*sizeof(size_t));
  rvs->label = CALLOC(vs->class_cnt, int);
  memcpy(rvs->label, vs->label, vs->class_cnt*sizeof(int));
  rvs->samples = CALLOC(vs->class_cnt, gsl_vector **);
  for(int class = 0; class < vs->class_cnt; class++) {
    rvs->samples[class] = CALLOC(vs->count[class], gsl_vector *);
  }

  round_pts.npos = CALLOC(n, int);
  round_pts.nneg = CALLOC(n, int);

  for(int i = 0; i < uf->n_sets; i++) {
    gsl_vector *avg = gsl_vector_calloc(vs->dimension);
    llnode *trav = uf->sets[i]->head;
    int npos = 0, nneg = 0;
    while(trav) {
      gsl_blas_daxpy(1, trav->pt, avg);
      if(vs->label[trav->class] == 1)
	npos++;
      else
	nneg++;
      trav = trav->next;
    }
    gsl_vector_scale(avg, 1./uf->sets[i]->n);
    trav = uf->sets[i]->head;
    while(trav) {
      rvs->samples[trav->class][trav->i] = gsl_vector_calloc(vs->dimension);
      gsl_vector_memcpy(rvs->samples[trav->class][trav->i], avg);
      trav = trav->next;
    }
    pts[i] = avg;
    round_pts.npos[i] = npos;
    round_pts.nneg[i] = nneg;
  }
  
  return (rounded_pts_t) {round_pts, rvs};
}

rounded_pts_t cluster_rounded_points_sep_classes(env_t *env, vsamples_t *vs, size_t n) {
  size_t d = env->samples->dimension;
  if(n > samples_total(env->samples)) {
    printf("ERR: too many round pts\n");
    return (rounded_pts_t) {0};
  }
  union_find_t *uf = create_union_find(samples_total(env->samples));
  make_sets_from_samples(uf, vs);
  uf_populate_dists(uf);
  
  while(uf->n_sets > n) {
    //find closest pair
    double min_dist = 1e101;
    int min_i, min_j;
    for(int i = 0; i < uf->n_sets; i++) {
      for(int j = i + 1; j < uf->n_sets; j++) {
	if(uf->sets[i]->head->class != uf->sets[j]->head->class)
	  continue;
	double dist = uf->sets[i]->dists[j];
	if(dist < min_dist) {
	  min_dist = dist;
	  min_i = i;
	  min_j = j;
	}
      }
    }
    uf_union(uf, uf->sets[min_i], uf->sets[min_j]);
  }
  
  round_pts_t round_pts;
  round_pts.n = n;
  gsl_vector **pts = CALLOC(n, gsl_vector *);
  round_pts.pts = pts;
  vsamples_t *rvs = malloc(sizeof(vsamples_t));
  rvs->dimension = d;
  rvs->class_cnt = vs->class_cnt;
  rvs->count = CALLOC(vs->class_cnt, size_t);
  memcpy(rvs->count, vs->count, vs->class_cnt*sizeof(size_t));
  rvs->label = CALLOC(vs->class_cnt, int);
  memcpy(rvs->label, vs->label, vs->class_cnt*sizeof(int));
  rvs->samples = CALLOC(vs->class_cnt, gsl_vector **);
  for(int class = 0; class < vs->class_cnt; class++) {
    rvs->samples[class] = CALLOC(vs->count[class], gsl_vector *);
  }

  round_pts.npos = CALLOC(n, int);
  round_pts.nneg = CALLOC(n, int);

  for(int i = 0; i < uf->n_sets; i++) {
    gsl_vector *avg = gsl_vector_calloc(vs->dimension);
    llnode *trav = uf->sets[i]->head;
    int npos = 0, nneg = 0;
    while(trav) {
      gsl_blas_daxpy(1, trav->pt, avg);
      //TODO: replace this with the npos and nneg struct fields already in the sets
      if(vs->label[trav->class] == 1)
	npos++;
      else
	nneg++;
      trav = trav->next;
    }
    gsl_vector_scale(avg, 1./uf->sets[i]->n);
    trav = uf->sets[i]->head;
    while(trav) {
      rvs->samples[trav->class][trav->i] = gsl_vector_calloc(vs->dimension);
      gsl_vector_memcpy(rvs->samples[trav->class][trav->i], avg);
      trav = trav->next;
    }
    pts[i] = avg;
    round_pts.npos[i] = npos;
    round_pts.nneg[i] = nneg;
  }
  
  return (rounded_pts_t) {round_pts, rvs};
}
