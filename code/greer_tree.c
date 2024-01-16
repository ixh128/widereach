#include "widereach.h"
#include "helper.h"

#include <math.h>
#include <memory.h>
#include <time.h>
#include <sys/time.h>
#include <assert.h>

#include <gsl/gsl_rng.h>
#include <gsl/gsl_linalg.h>

//to store the result of clasifying the dataset with a hyperplane, splitting samples into true positives, false positives, etc
//maybe it would be better to name these differently (based on "pos-pos" "pos-neg", "pos-zero", etc, so that the first word is always the class)
typedef struct class_res {
  int *tpos, *fpos, *tneg, *fneg, *pz, *nz;
  int ntpos, nfpos, ntneg, nfneg, npz, nnz;
  int nmiss;
} class_res;

//each node must store the hyperplane found at that node, its classification result, the "most recent" sample which the hyperplane is orthogonal to, and neighbors
//"most recent" means that you can traverse from a node to the root and get all of the samples to which the hyperplane is orthogonal
//the root node will have no loc
typedef struct tnode {
  gsl_vector *v;
  class_res res;
  sample_locator_t loc;
  int explored;
  struct tnode *parent, *l_sib, *r_sib, *l_child;
} tnode;

//stack for DFS (maybe put in another file)
typedef struct nodestack {
  tnode **nodes;
  size_t top, max; //top is the highest occupied index, or -1 if empty
} tnodestack;

tnodestack *create_stack() {
  tnodestack *out = CALLOC(1, tnodestack);
  out->max = 16;
  out->nodes = CALLOC(out->max, tnode *);
  out->top = -1;
  return out;
}

void stack_push(tnodestack *s, tnode *node) {
  if(s->top >= s->max - 1) {
    //if max is reached, double the size
    s = realloc(s, 2*s->max*sizeof(tnode *));
    if(!s) {
      printf("failed to realloc stack, capping at %ld\n", s->max);
    } else {
      s->max *= 2;
    }
  }
  s->nodes[++s->top] = node;
}

tnode *stack_pop(tnodestack *s) {
  return s->nodes[s->top--];
}

int stack_empty(tnodestack *s) {
  return s->top == -1;
}

class_res classify(vsamples_t *vs, gsl_vector *w) {
  assert(vs->class_cnt == 2);
  int *pos_res = CALLOC(vs->count[0], int);
  int *neg_res = CALLOC(vs->count[1], int);
  class_res out;
  for(size_t i = 0; i < vs->count[0]; i++) {
    double x;
    gsl_blas_ddot(vs->samples[0][i], w, &x);
    if(x > 0) {
      out.ntpos++;
      pos_res[i] = 1;
    } else if(x < 0) {
      out.nfneg++;
      pos_res[i] = -1;
    } else {
      out.npz++;
      pos_res[i] = 0;
    }
  }
  for(size_t i = 0; i < vs->count[1]; i++) {
    double x;
    gsl_blas_ddot(vs->samples[1][i], w, &x);
    if(x > 0) {
      out.nfpos++;
      neg_res[i] = 1;
    } else if(x < 0) {
      out.ntneg++;
      neg_res[i] = -1;
    } else {
      out.nnz++;
      pos_res[i] = 0;
    }
  }
  out.tpos = CALLOC(out.ntpos, int); int ktpos = 0;
  out.fpos = CALLOC(out.nfpos, int); int kfpos = 0;
  out.tneg = CALLOC(out.ntneg, int); int ktneg = 0;
  out.fneg = CALLOC(out.nfneg, int); int kfneg = 0;
  out.pz = CALLOC(out.npz, int); int kpz = 0;
  out.nz = CALLOC(out.nnz, int); int knz = 0;
  for(int i = 0; i < vs->count[0]; i++) {
    if(pos_res[i] == 1) out.tpos[ktpos++] = i;
    else if(pos_res[i] == -1) out.fneg[kfneg++] = i;
    else out.pz[kpz++] = i;
  }
  for(int i = 0; i < vs->count[0]; i++) {
    if(pos_res[i] == 1) out.fpos[kfpos++] = i;
    else if(pos_res[i] == -1) out.tneg[ktneg++] = i;
    else out.pz[knz++] = i;
  }

  out.nmiss = out.nfneg + out.nfpos;
  return out;
}

void free_class_res(class_res res) {
  free(res.tpos); free(res.fpos); free(res.tneg); free(res.fneg); free(res.pz); free(res.nz);
}

gsl_vector *obtain_perp(vsamples_t *vs, sample_locator_t *locs, int k) {
  //obtain a vector perpendicular to the k samples corresponding to locs
  //this does a QR factorization - maybe it's faster to do Gram-Schmidt or something else?
  //pretty similar to get_proj_basis - could be refactored
  size_t d = vs->dimension;
  gsl_matrix *A = gsl_matrix_calloc(d, k);
  for(int i = 0; i < k; i++) {
    gsl_matrix_set_col(A, i, vs->samples[locs[i].class][locs[i].index]);
  }
  gsl_matrix *T = gsl_matrix_alloc(d-1, d-1);
  gsl_linalg_QR_decomp_r(A, T);
  //final column of Q is the sole null-space vector of A, assuming A had full rank
  gsl_matrix *Q = gsl_matrix_alloc(d, d);
  gsl_matrix *R = gsl_matrix_alloc(d-1, d-1);
  gsl_linalg_QR_unpack_r(A, T, Q, R);

  gsl_vector *v = gsl_vector_alloc(d);
  gsl_matrix_get_col(v, Q, d-1);

  gsl_matrix_free(A);
  gsl_matrix_free(T);
  gsl_matrix_free(Q);
  gsl_matrix_free(R);

  return v;
}

void populate_node(vsamples_t *vs, tnode *node) {
  //runs classification on the node and creates its children. assumes tnode has a vector, maybe a loc, and maybe a parent, but nothing else
  node->res = classify(vs, node->v);
  
  tnode *trav;
}

void add_children_to_stack(tnodestack *s, tnode *curr) {
  
}

void gr_explore(vsamples_t *vs, sparse_vector_t *I, gsl_vector *v0, sparse_vector_t *A) {
  class_res res = classify(vs, v0);
  if(res.nfneg + res.nfpos > res.ntpos + res.ntneg) {
    gsl_vector_scale(v0, -1);
    res = classify(vs, v0); //TODO: this is unnecessary - just rearrange the existing items in res
  }
  if(res.nfneg + res.nfpos == 0) {
    printf("perfect vector found\n");
    return; //TODO - call displace
  }
  //actually, everything above here seems unnecessary - it could just be done in the DFS below

  size_t d = vs->dimension;
  tnode root = (tnode) {v0, res};

  tnodestack *s = create_stack();
  stack_push(s, &root);

  while(!stack_empty(s)) {
    tnode *curr = stack_pop(s);

    //find the vector v for curr if it doesn't already exist
    //classify v and store it in the node
    //then populate as below
    //doing it like this allows removing the above classification, since it'll be done here

    
    populate_node(vs, curr);
    add_children_to_stack(s, curr);
    
  }

  /*//trying to do DFS w/o a stack - maybe a bad idea
  populate_node(vs, trav);

  while(1) {
    //run everything on trav
    trav->explored = 1;
    
    
    if(trav->l_child) trav = trav->l_child;
    else if(trav->r_sib) trav = trav->r_sib;
    else if(trav->parent) {
      trav = trav->parent;
      
    }
    }*/
    
}

double *single_greer_run(env_t *env) {
  vsamples_t *vs = samples_to_vec(env->samples);

  gsl_vector *v0 = gsl_vector_calloc(vs->dimension);
  gsl_vector_set(v0, 0, 1);

  sparse_vector_t *I = sparse_vector_blank(samples_total(env->samples));
  for(int i = 0; i < samples_total(env->samples); i++) {
    append(I, i, 1);
  }

  //TODO
  return NULL;
}
