//
//  a LRU module that supports different obj size
//
//
//  LRU.c
//  libMimircache
//
//  Created by Juncheng on 12/4/18.
//  Copyright © 2018 Juncheng. All rights reserved.
//


#ifdef __cplusplus
extern "C" {
#endif


#include <assert.h>
#include "LRU.h"
#include "../utils/include/utilsInternal.h"


cache_t *LRU_init(guint64 size, obj_id_type_t obj_id_type, void *params) {
  cache_t *cache = cache_struct_init("LRU", size, obj_id_type);
  cache->cache_params = g_new0(LRU_params_t, 1);
  LRU_params_t *LRU_params = (LRU_params_t *) (cache->cache_params);
  LRU_params->hashtable = create_hash_table_with_obj_id_type(obj_id_type, NULL, NULL, g_free, NULL);
  LRU_params->list = g_queue_new();
  return cache;
}

void LRU_free(cache_t *cache) {
  LRU_params_t *LRU_params = (LRU_params_t *) (cache->cache_params);
  g_hash_table_destroy(LRU_params->hashtable);
  g_queue_free_full(LRU_params->list, cache_obj_destroyer);
  cache_struct_free(cache);
}

gboolean LRU_check(cache_t *cache, request_t *req) {
  LRU_params_t *LRU_params = (LRU_params_t *) (cache->cache_params);
  return g_hash_table_contains(LRU_params->hashtable, req->obj_id_ptr);
}

gboolean LRU_get(cache_t *cache, request_t *req) {
  gboolean found_in_cache = LRU_check(cache, req);
  if (req->obj_size <= cache->core->size) {
    if (found_in_cache)
      _LRU_update(cache, req);
    else
      _LRU_insert(cache, req);

    while (cache->core->used_size > cache->core->size)
      _LRU_evict(cache, req);
  } else {
    WARNING("req %lld: obj size %ld larger than cache size %ld\n", (long long)cache->core->req_cnt,
            (long) req->obj_size, (long) cache->core->size);
  }
  cache->core->req_cnt += 1;
  return found_in_cache;
}

void _LRU_insert(cache_t *cache, request_t *req) {
  LRU_params_t *LRU_params = (LRU_params_t *) (cache->cache_params);
  cache->core->used_size += req->obj_size;
  cache_obj_t* cache_obj = create_cache_obj_from_req(req);

  GList *node = g_list_alloc();
  node->data = cache_obj;
  g_queue_push_tail_link(LRU_params->list, node);
  g_hash_table_insert(LRU_params->hashtable, cache_obj->obj_id_ptr, (gpointer) node);
}

void _LRU_update(cache_t *cache, request_t *req) {
  LRU_params_t *LRU_params = (LRU_params_t *) (cache->cache_params);
  GList *node = (GList *) g_hash_table_lookup(LRU_params->hashtable, req->obj_id_ptr);

  cache_obj_t *cache_obj = node->data;
  assert(cache->core->used_size >= cache_obj->size);
  cache->core->used_size -= cache_obj->size;
  cache->core->used_size += req->obj_size;
  update_cache_obj(cache_obj, req);
  g_queue_unlink(LRU_params->list, node);
  g_queue_push_tail_link(LRU_params->list, node);
}

cache_obj_t *LRU_get_cached_obj(cache_t *cache, request_t *req) {
  LRU_params_t *LRU_params = (LRU_params_t *) (cache->cache_params);
  GList *node = (GList *) g_hash_table_lookup(LRU_params->hashtable, req->obj_id_ptr);
  cache_obj_t *cache_obj = node->data;
  return cache_obj;
}

void _LRU_evict(cache_t *cache, request_t *req) {
  LRU_params_t *LRU_params = (LRU_params_t *) (cache->cache_params);
  cache_obj_t *cache_obj = (cache_obj_t *) g_queue_pop_head(LRU_params->list);

  assert(cache->core->used_size >= cache_obj->size);
  cache->core->used_size -= cache_obj->size;
  g_hash_table_remove(LRU_params->hashtable, (gconstpointer) cache_obj->obj_id_ptr);
  destroy_cache_obj(cache_obj);
}

gpointer _LRU_evict_with_return(cache_t *cache, request_t *req) {
  LRU_params_t *LRU_params = (LRU_params_t *) (cache->cache_params);
  cache_obj_t *cache_obj = g_queue_pop_head(LRU_params->list);
  assert(cache->core->used_size >= cache_obj->size);
  cache->core->used_size -= cache_obj->size;
  gpointer evicted_key = cache_obj->obj_id_ptr;
  if (req->obj_id_type == OBJ_ID_STR) {
    evicted_key = (gpointer) g_strdup((gchar *) (cache_obj->obj_id_ptr));
  }
  g_hash_table_remove(LRU_params->hashtable, (gconstpointer) (cache_obj->obj_id_ptr));
  destroy_cache_obj(cache_obj);
  return evicted_key;
}


void LRU_remove_obj(cache_t *cache, gpointer obj_id_ptr) {
  LRU_params_t *LRU_params = (LRU_params_t *) (cache->cache_params);

  GList *node = (GList *) g_hash_table_lookup(LRU_params->hashtable, obj_id_ptr);
  if (!node) {
    ERROR("obj to remove is not in the cache\n");
    abort();
  }
  cache_obj_t* cache_obj = (cache_obj_t *) (node->data);
  assert(cache->core->used_size >= cache_obj->size);
  cache->core->used_size -= ((cache_obj_t *) (node->data))->size;
  g_queue_delete_link(LRU_params->list, (GList *) node);
  g_hash_table_remove(LRU_params->hashtable, obj_id_ptr);
  destroy_cache_obj(cache_obj);
}


//GHashTable *LRU_get_objmap(cache_t *cache) {
//  LRU_params_t *LRU_params = (LRU_params_t *) (cache->cache_params);
//  return LRU_params->hashtable;
//}

#ifdef __cplusplus
extern "C" {
#endif
