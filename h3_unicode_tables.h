#ifndef H3_UNICODE_TABLES_H
#define H3_UNICODE_TABLES_H

#include <stddef.h>
#include <stdint.h>

/* Sorted, non-overlapping inclusive codepoint ranges. */
typedef struct {
    uint32_t first;
    uint32_t last;
} h3_unicode_range;

/* Sorted, non-overlapping ranges sharing a non-zero canonical
 * combining class. */
typedef struct {
    uint32_t first;
    uint32_t last;
    uint8_t ccc;
} h3_unicode_ccc_range;

/* Canonical (non-compatibility) decomposition, sorted by codepoint.
 * right is 0 for singleton decompositions. BMP only; Hangul is
 * algorithmic. */
typedef struct {
    uint32_t codepoint;
    uint32_t left;
    uint32_t right;
} h3_unicode_decompose_entry;

/* Composition pair (left, right) -> composite, sorted by (left, right).
 * BMP only; Hangul is algorithmic. */
typedef struct {
    uint32_t left;
    uint32_t right;
    uint32_t composite;
} h3_unicode_compose_entry;

extern const h3_unicode_range h3_unicode_letters[];
extern const size_t h3_unicode_letters_count;
extern const h3_unicode_range h3_unicode_numbers[];
extern const size_t h3_unicode_numbers_count;
extern const h3_unicode_ccc_range h3_unicode_ccc[];
extern const size_t h3_unicode_ccc_count;
extern const h3_unicode_decompose_entry h3_unicode_decompose[];
extern const size_t h3_unicode_decompose_count;
extern const h3_unicode_compose_entry h3_unicode_compose[];
extern const size_t h3_unicode_compose_count;

#endif
