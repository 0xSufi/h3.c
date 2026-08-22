/* Portable C11 implementation of the Qwen2 byte-level BPE tokenizer.
 * This is a semantics-preserving port of h3_tokenizer.m (ObjC/ICU) for
 * platforms without Foundation or ICU. Unicode classification and NFC
 * normalization come from the generated tables in h3_unicode_tables.c. */

#include "h3_tokenizer.h"

#include "h3_unicode_tables.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define H3_BYTE_DECODER_SIZE 324

/* ---------- small utilities ---------- */

static void h3_tok_error(char *error, size_t size, const char *message) {
    if (error && size) snprintf(error, size, "%s", message);
}

static char *h3_strdup(const char *text) {
    size_t length = strlen(text);
    char *copy = malloc(length + 1);
    if (copy) memcpy(copy, text, length + 1);
    return copy;
}

static char *h3_strndup(const char *text, size_t length) {
    char *copy = malloc(length + 1);
    if (copy) {
        memcpy(copy, text, length);
        copy[length] = '\0';
    }
    return copy;
}

/* ---------- UTF-8 ---------- */

/* Strict UTF-8 validation, matching NSString's stringWithUTF8String:
 * rejects overlong forms, surrogates and values above U+10FFFF. */
static int h3_utf8_valid(const char *text, size_t length) {
    size_t index = 0;
    while (index < length) {
        unsigned char lead = (unsigned char)text[index];
        uint32_t value;
        size_t width;
        if (lead < 0x80) {
            index++;
            continue;
        }
        if (lead >= 0xc2 && lead <= 0xdf) {
            value = lead & 0x1f;
            width = 2;
        } else if (lead >= 0xe0 && lead <= 0xef) {
            value = lead & 0x0f;
            width = 3;
        } else if (lead >= 0xf0 && lead <= 0xf4) {
            value = lead & 0x07;
            width = 4;
        } else {
            return 0;
        }
        if (index + width > length) return 0;
        for (size_t offset = 1; offset < width; offset++) {
            unsigned char trail = (unsigned char)text[index + offset];
            if ((trail & 0xc0) != 0x80) return 0;
            value = (value << 6) | (uint32_t)(trail & 0x3f);
        }
        if (value < (width == 2 ? UINT32_C(0x80)
                                : width == 3 ? UINT32_C(0x800)
                                             : UINT32_C(0x10000)))
            return 0;
        if (value >= 0xd800 && value <= 0xdfff) return 0;
        if (value > 0x10ffff) return 0;
        index += width;
    }
    return 1;
}

/* Decode one codepoint. Callers only pass valid UTF-8; malformed input
 * degrades to one byte at a time. */
static uint32_t h3_utf8_decode(const char *text, size_t length,
                               size_t *width) {
    unsigned char lead = (unsigned char)text[0];
    if (lead < 0x80) {
        *width = 1;
        return lead;
    }
    if (lead >= 0xc2 && lead <= 0xdf && length >= 2 &&
        ((unsigned char)text[1] & 0xc0) == 0x80) {
        *width = 2;
        return ((uint32_t)(lead & 0x1f) << 6) |
               (uint32_t)((unsigned char)text[1] & 0x3f);
    }
    if (lead >= 0xe0 && lead <= 0xef && length >= 3 &&
        ((unsigned char)text[1] & 0xc0) == 0x80 &&
        ((unsigned char)text[2] & 0xc0) == 0x80) {
        *width = 3;
        return ((uint32_t)(lead & 0x0f) << 12) |
               ((uint32_t)((unsigned char)text[1] & 0x3f) << 6) |
               (uint32_t)((unsigned char)text[2] & 0x3f);
    }
    if (lead >= 0xf0 && lead <= 0xf4 && length >= 4 &&
        ((unsigned char)text[1] & 0xc0) == 0x80 &&
        ((unsigned char)text[2] & 0xc0) == 0x80 &&
        ((unsigned char)text[3] & 0xc0) == 0x80) {
        *width = 4;
        return ((uint32_t)(lead & 0x07) << 18) |
               ((uint32_t)((unsigned char)text[1] & 0x3f) << 12) |
               ((uint32_t)((unsigned char)text[2] & 0x3f) << 6) |
               (uint32_t)((unsigned char)text[3] & 0x3f);
    }
    *width = 1;
    return lead;
}

static size_t h3_utf8_encode(uint32_t value, char out[4]) {
    if (value < 0x80) {
        out[0] = (char)value;
        return 1;
    }
    if (value < 0x800) {
        out[0] = (char)(0xc0 | (value >> 6));
        out[1] = (char)(0x80 | (value & 0x3f));
        return 2;
    }
    if (value < 0x10000) {
        out[0] = (char)(0xe0 | (value >> 12));
        out[1] = (char)(0x80 | ((value >> 6) & 0x3f));
        out[2] = (char)(0x80 | (value & 0x3f));
        return 3;
    }
    out[0] = (char)(0xf0 | (value >> 18));
    out[1] = (char)(0x80 | ((value >> 12) & 0x3f));
    out[2] = (char)(0x80 | ((value >> 6) & 0x3f));
    out[3] = (char)(0x80 | (value & 0x3f));
    return 4;
}

/* ---------- Unicode classification ---------- */

static int h3_range_has(const h3_unicode_range *ranges, size_t count,
                        uint32_t value) {
    size_t low = 0;
    size_t high = count;
    while (low < high) {
        size_t middle = low + (high - low) / 2;
        if (value < ranges[middle].first) high = middle;
        else if (value > ranges[middle].last) low = middle + 1;
        else return 1;
    }
    return 0;
}

static int h3_letter(uint32_t value) {
    return h3_range_has(h3_unicode_letters, h3_unicode_letters_count, value);
}

static int h3_number(uint32_t value) {
    return h3_range_has(h3_unicode_numbers, h3_unicode_numbers_count, value);
}

static int h3_space(uint32_t value) {
    /* ICU u_isUWhiteSpace == Unicode White_Space property, a short fixed
     * list: 0009-000D, 0020, 0085, 00A0, 1680, 2000-200A, 2028, 2029,
     * 202F, 205F, 3000. The 0x1C-0x1F extension matches h3_tokenizer.m. */
    if (value >= 0x1c && value <= 0x1f) return 1;
    return (value >= 0x09 && value <= 0x0d) || value == 0x20 ||
           value == 0x85 || value == 0xa0 || value == 0x1680 ||
           (value >= 0x2000 && value <= 0x200a) || value == 0x2028 ||
           value == 0x2029 || value == 0x202f || value == 0x205f ||
           value == 0x3000;
}

static uint8_t h3_combining_class(uint32_t value) {
    size_t low = 0;
    size_t high = h3_unicode_ccc_count;
    while (low < high) {
        size_t middle = low + (high - low) / 2;
        if (value < h3_unicode_ccc[middle].first) high = middle;
        else if (value > h3_unicode_ccc[middle].last) low = middle + 1;
        else return h3_unicode_ccc[middle].ccc;
    }
    return 0;
}

/* ---------- NFC ---------- */

#define H3_HANGUL_SBASE UINT32_C(0xac00)
#define H3_HANGUL_LBASE UINT32_C(0x1100)
#define H3_HANGUL_VBASE UINT32_C(0x1161)
#define H3_HANGUL_TBASE UINT32_C(0x11a7)
#define H3_HANGUL_LCOUNT UINT32_C(19)
#define H3_HANGUL_VCOUNT UINT32_C(21)
#define H3_HANGUL_TCOUNT UINT32_C(28)
#define H3_HANGUL_NCOUNT (H3_HANGUL_VCOUNT * H3_HANGUL_TCOUNT)
#define H3_HANGUL_SCOUNT (H3_HANGUL_LCOUNT * H3_HANGUL_NCOUNT)

static const h3_unicode_decompose_entry *h3_decompose_lookup(uint32_t value) {
    size_t low = 0;
    size_t high = h3_unicode_decompose_count;
    while (low < high) {
        size_t middle = low + (high - low) / 2;
        if (value < h3_unicode_decompose[middle].codepoint) high = middle;
        else if (value > h3_unicode_decompose[middle].codepoint) low = middle + 1;
        else return &h3_unicode_decompose[middle];
    }
    return NULL;
}

static uint32_t h3_compose_lookup(uint32_t left, uint32_t right) {
    if (left >= H3_HANGUL_LBASE && left < H3_HANGUL_LBASE + H3_HANGUL_LCOUNT &&
        right >= H3_HANGUL_VBASE && right < H3_HANGUL_VBASE + H3_HANGUL_VCOUNT) {
        return H3_HANGUL_SBASE +
               ((left - H3_HANGUL_LBASE) * H3_HANGUL_VCOUNT +
                (right - H3_HANGUL_VBASE)) * H3_HANGUL_TCOUNT;
    }
    if (left >= H3_HANGUL_SBASE && left < H3_HANGUL_SBASE + H3_HANGUL_SCOUNT &&
        (left - H3_HANGUL_SBASE) % H3_HANGUL_TCOUNT == 0 &&
        right > H3_HANGUL_TBASE && right < H3_HANGUL_TBASE + H3_HANGUL_TCOUNT) {
        return left + (right - H3_HANGUL_TBASE);
    }
    size_t low = 0;
    size_t high = h3_unicode_compose_count;
    while (low < high) {
        size_t middle = low + (high - low) / 2;
        const h3_unicode_compose_entry *entry = &h3_unicode_compose[middle];
        if (left < entry->left ||
            (left == entry->left && right < entry->right)) {
            high = middle;
        } else if (left == entry->left && right == entry->right) {
            return entry->composite;
        } else {
            low = middle + 1;
        }
    }
    return 0;
}

static int h3_nfc_push(uint32_t **buffer, size_t *used, size_t *capacity,
                       uint32_t value) {
    if (*used == *capacity) {
        size_t next = *capacity ? *capacity * 2 : 16;
        uint32_t *grown = realloc(*buffer, next * sizeof(*grown));
        if (!grown) return 0;
        *buffer = grown;
        *capacity = next;
    }
    (*buffer)[(*used)++] = value;
    return 1;
}

/* Full recursive canonical decomposition (Hangul is algorithmic). */
static int h3_nfc_decompose(uint32_t value, uint32_t **buffer, size_t *used,
                            size_t *capacity) {
    if (value >= H3_HANGUL_SBASE &&
        value < H3_HANGUL_SBASE + H3_HANGUL_SCOUNT) {
        uint32_t syllable = value - H3_HANGUL_SBASE;
        if (!h3_nfc_push(buffer, used, capacity,
                         H3_HANGUL_LBASE + syllable / H3_HANGUL_NCOUNT))
            return 0;
        if (!h3_nfc_push(buffer, used, capacity,
                         H3_HANGUL_VBASE +
                             (syllable % H3_HANGUL_NCOUNT) / H3_HANGUL_TCOUNT))
            return 0;
        if (syllable % H3_HANGUL_TCOUNT)
            return h3_nfc_push(buffer, used, capacity,
                               H3_HANGUL_TBASE + syllable % H3_HANGUL_TCOUNT);
        return 1;
    }
    const h3_unicode_decompose_entry *entry = h3_decompose_lookup(value);
    if (entry) {
        if (!h3_nfc_decompose(entry->left, buffer, used, capacity)) return 0;
        if (entry->right)
            return h3_nfc_decompose(entry->right, buffer, used, capacity);
        return 1;
    }
    return h3_nfc_push(buffer, used, capacity, value);
}

/* NFC normalize valid UTF-8; returns a malloc'd NUL-terminated string. */
static char *h3_nfc(const char *text, size_t length, size_t *out_length) {
    uint32_t *buffer = NULL;
    size_t used = 0;
    size_t capacity = 0;
    size_t index = 0;
    while (index < length) {
        size_t width;
        uint32_t value = h3_utf8_decode(text + index, length - index, &width);
        index += width;
        if (!h3_nfc_decompose(value, &buffer, &used, &capacity)) {
            free(buffer);
            return NULL;
        }
    }
    /* Canonical ordering: stable reorder by combining class. */
    for (size_t outer = 1; outer < used; outer++) {
        uint8_t ccc = h3_combining_class(buffer[outer]);
        size_t inner = outer;
        uint32_t value;
        if (!ccc) continue;
        value = buffer[outer];
        while (inner > 0) {
            uint8_t previous = h3_combining_class(buffer[inner - 1]);
            if (!previous || previous <= ccc) break;
            buffer[inner] = buffer[inner - 1];
            inner--;
        }
        buffer[inner] = value;
    }
    /* Composition. */
    size_t written = 0;
    size_t starter = 0;
    int have_starter = 0;
    uint8_t last_ccc = 0;
    for (size_t read = 0; read < used; read++) {
        uint8_t ccc = h3_combining_class(buffer[read]);
        uint32_t composite = 0;
        if (have_starter && (last_ccc < ccc || last_ccc == 0) &&
            (composite = h3_compose_lookup(buffer[starter], buffer[read]))) {
            buffer[starter] = composite;
            continue;
        }
        if (!ccc) {
            starter = written;
            have_starter = 1;
        }
        buffer[written++] = buffer[read];
        last_ccc = ccc;
    }
    used = written;
    char *result = malloc(used * 4 + 1);
    if (!result) {
        free(buffer);
        return NULL;
    }
    size_t produced = 0;
    for (index = 0; index < used; index++) {
        char encoded[4];
        size_t width = h3_utf8_encode(buffer[index], encoded);
        memcpy(result + produced, encoded, width);
        produced += width;
    }
    result[produced] = '\0';
    free(buffer);
    if (out_length) *out_length = produced;
    return result;
}

/* ---------- string -> value map (open addressing) ---------- */

typedef struct {
    char *key;       /* owned */
    uint32_t value;  /* token id or merge rank */
    uint32_t *ids;   /* owned, BPE cache only */
    size_t count;    /* BPE cache only */
} h3_slot;

typedef struct {
    h3_slot *slots;
    size_t used;
    size_t capacity;
} h3_map;

static uint64_t h3_hash(const char *key) {
    uint64_t hash = UINT64_C(1469598103934665603);
    while (*key) {
        hash ^= (unsigned char)*key++;
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static int h3_map_reserve(h3_map *map) {
    if (map->capacity && (map->used + 1) * 10 <= map->capacity * 7) return 1;
    size_t capacity = map->capacity ? map->capacity * 2 : 1024;
    h3_slot *slots = calloc(capacity, sizeof(*slots));
    if (!slots) return 0;
    for (size_t index = 0; index < map->capacity; index++) {
        h3_slot entry = map->slots[index];
        if (!entry.key) continue;
        size_t target = (size_t)(h3_hash(entry.key) & (uint64_t)(capacity - 1));
        while (slots[target].key) target = (target + 1) & (capacity - 1);
        slots[target] = entry;
    }
    free(map->slots);
    map->slots = slots;
    map->capacity = capacity;
    return 1;
}

static h3_slot *h3_map_probe(h3_map *map, const char *key) {
    size_t index = (size_t)(h3_hash(key) & (uint64_t)(map->capacity - 1));
    while (map->slots[index].key && strcmp(map->slots[index].key, key))
        index = (index + 1) & (map->capacity - 1);
    return &map->slots[index];
}

static const h3_slot *h3_map_find(const h3_map *map, const char *key) {
    if (!map->capacity) return NULL;
    size_t index = (size_t)(h3_hash(key) & (uint64_t)(map->capacity - 1));
    while (map->slots[index].key) {
        if (!strcmp(map->slots[index].key, key)) return &map->slots[index];
        index = (index + 1) & (map->capacity - 1);
    }
    return NULL;
}

static int h3_map_get_u32(const h3_map *map, const char *key,
                          uint32_t *value) {
    const h3_slot *slot = h3_map_find(map, key);
    if (!slot) return 0;
    *value = slot->value;
    return 1;
}

/* Takes ownership of key (freed on failure or duplicate). */
static int h3_map_put_u32(h3_map *map, char *key, uint32_t value) {
    if (!h3_map_reserve(map)) {
        free(key);
        return 0;
    }
    h3_slot *slot = h3_map_probe(map, key);
    if (slot->key) {
        free(key);
        slot->value = value;
        return 1;
    }
    slot->key = key;
    slot->value = value;
    map->used++;
    return 1;
}

/* Takes ownership of key; copies ids. */
static int h3_map_put_ids(h3_map *map, char *key, const uint32_t *ids,
                          size_t count) {
    uint32_t *copy = malloc(count * sizeof(*copy));
    if (!copy) {
        free(key);
        return 0;
    }
    memcpy(copy, ids, count * sizeof(*copy));
    if (!h3_map_reserve(map)) {
        free(key);
        free(copy);
        return 0;
    }
    h3_slot *slot = h3_map_probe(map, key);
    if (slot->key) {
        free(key);
        free(slot->ids);
    } else {
        slot->key = key;
        map->used++;
    }
    slot->ids = copy;
    slot->count = count;
    return 1;
}

static void h3_map_clear(h3_map *map) {
    for (size_t index = 0; index < map->capacity; index++) {
        free(map->slots[index].key);
        free(map->slots[index].ids);
    }
    free(map->slots);
    map->slots = NULL;
    map->used = 0;
    map->capacity = 0;
}

/* ---------- tokenizer state ---------- */

struct h3_tokenizer {
    h3_map vocab;   /* byte-level symbol -> token id */
    h3_map merges;  /* pair key (left + U+FFFF + right) -> merge rank */
    h3_map added;   /* added token content -> token id */
    char **alternatives; /* added contents, longest first; borrowed */
    size_t alternative_count;
    char **inverse_vocab; /* id -> symbol; borrowed from vocab keys */
    char **inverse_added; /* id -> content; borrowed from added keys */
    size_t inverse_count;
    h3_map cache;   /* byte-encoded piece -> token id array */
    int16_t byte_decoder[H3_BYTE_DECODER_SIZE];
    uint32_t byte_encoder[256];
};

/* ---------- minimal JSON cursor ---------- */

typedef struct {
    const char *at;
    const char *end;
    char error[128];
} h3_json;

static int h3_json_fail(h3_json *json, const char *message) {
    if (!json->error[0]) snprintf(json->error, sizeof(json->error), "%s", message);
    return 0;
}

static int h3_json_syntax(h3_json *json) {
    return h3_json_fail(json, "invalid tokenizer JSON");
}

static void h3_json_ws(h3_json *json) {
    while (json->at < json->end &&
           (*json->at == ' ' || *json->at == '\t' || *json->at == '\n' ||
            *json->at == '\r'))
        json->at++;
}

static int h3_json_peek(h3_json *json, char expected) {
    h3_json_ws(json);
    return json->at < json->end && *json->at == expected;
}

static int h3_json_take(h3_json *json, char expected) {
    if (!h3_json_peek(json, expected)) return 0;
    json->at++;
    return 1;
}

static int h3_json_expect(h3_json *json, char expected) {
    if (h3_json_take(json, expected)) return 1;
    return h3_json_syntax(json);
}

static int h3_json_hex4(h3_json *json, uint32_t *unit) {
    uint32_t value = 0;
    if (json->end - json->at < 4) return 0;
    for (int index = 0; index < 4; index++) {
        char digit = *json->at++;
        value <<= 4;
        if (digit >= '0' && digit <= '9') value |= (uint32_t)(digit - '0');
        else if (digit >= 'a' && digit <= 'f') value |= (uint32_t)(digit - 'a' + 10);
        else if (digit >= 'A' && digit <= 'F') value |= (uint32_t)(digit - 'A' + 10);
        else return 0;
    }
    *unit = value;
    return 1;
}

static int h3_json_append(char **buffer, size_t *used, size_t *capacity,
                          const char *bytes, size_t count) {
    if (*used + count + 1 > *capacity) {
        size_t next = *capacity ? *capacity * 2 : 32;
        while (*used + count + 1 > next) next *= 2;
        char *grown = realloc(*buffer, next);
        if (!grown) return 0;
        *buffer = grown;
        *capacity = next;
    }
    memcpy(*buffer + *used, bytes, count);
    *used += count;
    return 1;
}

/* Parse a JSON string into a malloc'd NUL-terminated UTF-8 string. */
static char *h3_json_string(h3_json *json) {
    h3_json_ws(json);
    if (json->at >= json->end || *json->at != '"') {
        h3_json_syntax(json);
        return NULL;
    }
    json->at++;
    char *buffer = NULL;
    size_t used = 0;
    size_t capacity = 0;
    for (;;) {
        if (json->at >= json->end) {
            h3_json_syntax(json);
            free(buffer);
            return NULL;
        }
        unsigned char byte = (unsigned char)*json->at++;
        if (byte == '"') break;
        if (byte == '\\') {
            char unit;
            if (json->at >= json->end) {
                h3_json_syntax(json);
                free(buffer);
                return NULL;
            }
            char escape = *json->at++;
            switch (escape) {
            case '"': case '\\': case '/': unit = escape; break;
            case 'b': unit = '\b'; break;
            case 'f': unit = '\f'; break;
            case 'n': unit = '\n'; break;
            case 'r': unit = '\r'; break;
            case 't': unit = '\t'; break;
            case 'u': {
                uint32_t first;
                uint32_t codepoint;
                if (!h3_json_hex4(json, &first)) {
                    h3_json_syntax(json);
                    free(buffer);
                    return NULL;
                }
                if (first >= 0xd800 && first <= 0xdbff) {
                    uint32_t second;
                    if (json->end - json->at < 6 || json->at[0] != '\\' ||
                        json->at[1] != 'u') {
                        h3_json_syntax(json);
                        free(buffer);
                        return NULL;
                    }
                    json->at += 2;
                    if (!h3_json_hex4(json, &second) || second < 0xdc00 ||
                        second > 0xdfff) {
                        h3_json_syntax(json);
                        free(buffer);
                        return NULL;
                    }
                    codepoint = UINT32_C(0x10000) +
                                ((first - 0xd800) << 10) + (second - 0xdc00);
                } else if (first >= 0xdc00 && first <= 0xdfff) {
                    h3_json_syntax(json);
                    free(buffer);
                    return NULL;
                } else {
                    codepoint = first;
                }
                {
                    char encoded[4];
                    size_t width = h3_utf8_encode(codepoint, encoded);
                    if (!h3_json_append(&buffer, &used, &capacity, encoded,
                                        width)) {
                        h3_json_fail(json, "out of memory parsing tokenizer JSON");
                        free(buffer);
                        return NULL;
                    }
                }
                continue;
            }
            default:
                h3_json_syntax(json);
                free(buffer);
                return NULL;
            }
            if (!h3_json_append(&buffer, &used, &capacity, &unit, 1)) {
                h3_json_fail(json, "out of memory parsing tokenizer JSON");
                free(buffer);
                return NULL;
            }
            continue;
        }
        if (byte < 0x20) {
            h3_json_syntax(json);
            free(buffer);
            return NULL;
        }
        if (!h3_json_append(&buffer, &used, &capacity, (const char *)&byte,
                            1)) {
            h3_json_fail(json, "out of memory parsing tokenizer JSON");
            free(buffer);
            return NULL;
        }
    }
    if (!h3_json_append(&buffer, &used, &capacity, "", 1)) {
        /* Only reachable on the very first allocation failing. */
        h3_json_fail(json, "out of memory parsing tokenizer JSON");
        free(buffer);
        return NULL;
    }
    buffer[used] = '\0';
    return buffer;
}

static int h3_json_uint(h3_json *json, uint64_t *result) {
    uint64_t value = 0;
    const char *start;
    h3_json_ws(json);
    start = json->at;
    while (json->at < json->end && *json->at >= '0' && *json->at <= '9') {
        unsigned digit = (unsigned)(*json->at - '0');
        if (value > (UINT64_MAX - 9) / 10) return h3_json_syntax(json);
        value = value * 10 + digit;
        json->at++;
    }
    if (json->at == start) return h3_json_syntax(json);
    *result = value;
    return 1;
}

static int h3_json_null(h3_json *json) {
    h3_json_ws(json);
    if (json->end - json->at >= 4 && !memcmp(json->at, "null", 4)) {
        json->at += 4;
        return 1;
    }
    return h3_json_syntax(json);
}

static int h3_json_bool(h3_json *json, int *result) {
    h3_json_ws(json);
    if (json->end - json->at >= 4 && !memcmp(json->at, "true", 4)) {
        json->at += 4;
        *result = 1;
        return 1;
    }
    if (json->end - json->at >= 5 && !memcmp(json->at, "false", 5)) {
        json->at += 5;
        *result = 0;
        return 1;
    }
    return h3_json_syntax(json);
}

static int h3_json_delimiter(char byte) {
    return byte == ' ' || byte == '\t' || byte == '\n' || byte == '\r' ||
           byte == ',' || byte == ']' || byte == '}';
}

static int h3_json_skip(h3_json *json) {
    h3_json_ws(json);
    if (json->at >= json->end) return h3_json_syntax(json);
    if (*json->at == '"') {
        char *value = h3_json_string(json);
        if (!value) return 0;
        free(value);
        return 1;
    }
    if (*json->at == '{') {
        json->at++;
        if (!h3_json_peek(json, '}')) {
            for (;;) {
                char *key = h3_json_string(json);
                if (!key) return 0;
                free(key);
                if (!h3_json_expect(json, ':')) return 0;
                if (!h3_json_skip(json)) return 0;
                if (h3_json_take(json, ',')) continue;
                break;
            }
        }
        return h3_json_expect(json, '}');
    }
    if (*json->at == '[') {
        json->at++;
        if (!h3_json_peek(json, ']')) {
            for (;;) {
                if (!h3_json_skip(json)) return 0;
                if (h3_json_take(json, ',')) continue;
                break;
            }
        }
        return h3_json_expect(json, ']');
    }
    {
        const char *start = json->at;
        while (json->at < json->end && !h3_json_delimiter(*json->at))
            json->at++;
        if (json->at == start) return h3_json_syntax(json);
        return 1;
    }
}

/* ---------- tokenizer.json parsing ---------- */

static int h3_parse_vocab(h3_json *json, h3_tokenizer *tokenizer,
                          uint32_t *maximum_id) {
    if (!h3_json_expect(json, '{')) return 0;
    if (!h3_json_peek(json, '}')) {
        for (;;) {
            char *key = h3_json_string(json);
            uint64_t id;
            if (!key) return 0;
            if (!h3_json_expect(json, ':') || !h3_json_uint(json, &id)) {
                free(key);
                return 0;
            }
            if (id > UINT32_MAX) {
                free(key);
                return h3_json_fail(json, "unexpected tokenizer specification");
            }
            if ((uint32_t)id > *maximum_id) *maximum_id = (uint32_t)id;
            if (!h3_map_put_u32(&tokenizer->vocab, key, (uint32_t)id))
                return h3_json_fail(json, "out of memory loading tokenizer");
            if (h3_json_take(json, ',')) continue;
            break;
        }
    }
    return h3_json_expect(json, '}');
}

static char *h3_pair_key(const char *left, size_t left_length,
                         const char *right, size_t right_length) {
    /* Merge pairs are keyed by left + U+FFFF + right (UTF-8 EF BF BF). */
    char *key = malloc(left_length + 3 + right_length + 1);
    if (!key) return NULL;
    memcpy(key, left, left_length);
    memcpy(key + left_length, "\xef\xbf\xbf", 3);
    memcpy(key + left_length + 3, right, right_length);
    key[left_length + 3 + right_length] = '\0';
    return key;
}

static int h3_parse_merges(h3_json *json, h3_tokenizer *tokenizer) {
    uint32_t rank = 0;
    if (!h3_json_expect(json, '[')) return 0;
    if (!h3_json_peek(json, ']')) {
        for (;;) {
            char *key = NULL;
            h3_json_ws(json);
            if (h3_json_peek(json, '"')) {
                char *entry = h3_json_string(json);
                char *separator;
                if (!entry) return 0;
                separator = strchr(entry, ' ');
                if (!separator) {
                    free(entry);
                    return h3_json_fail(json, "invalid tokenizer merge");
                }
                key = h3_pair_key(entry, (size_t)(separator - entry),
                                  separator + 1, strlen(separator + 1));
                free(entry);
            } else if (h3_json_peek(json, '[')) {
                char *left;
                char *right;
                if (!h3_json_expect(json, '[')) return 0;
                left = h3_json_string(json);
                if (!left) return 0;
                if (!h3_json_expect(json, ',')) {
                    free(left);
                    return 0;
                }
                right = h3_json_string(json);
                if (!right || !h3_json_expect(json, ']')) {
                    free(left);
                    free(right);
                    return 0;
                }
                key = h3_pair_key(left, strlen(left), right, strlen(right));
                free(left);
                free(right);
            } else {
                return h3_json_fail(json, "invalid tokenizer merge");
            }
            if (!key)
                return h3_json_fail(json, "out of memory loading tokenizer");
            if (!h3_map_put_u32(&tokenizer->merges, key, rank++))
                return h3_json_fail(json, "out of memory loading tokenizer");
            if (h3_json_take(json, ',')) continue;
            break;
        }
    }
    return h3_json_expect(json, ']');
}

static int h3_parse_added(h3_json *json, h3_tokenizer *tokenizer,
                          uint32_t *maximum_id) {
    if (!h3_json_expect(json, '[')) return 0;
    if (!h3_json_peek(json, ']')) {
        for (;;) {
            int have_id = 0;
            uint64_t id = 0;
            char *content = NULL;
            if (!h3_json_expect(json, '{')) return 0;
            if (!h3_json_peek(json, '}')) {
                for (;;) {
                    char *key = h3_json_string(json);
                    if (!key) {
                        free(content);
                        return 0;
                    }
                    if (!h3_json_expect(json, ':')) {
                        free(key);
                        free(content);
                        return 0;
                    }
                    if (!strcmp(key, "id")) {
                        if (!h3_json_uint(json, &id)) {
                            free(key);
                            free(content);
                            return 0;
                        }
                        have_id = 1;
                    } else if (!strcmp(key, "content")) {
                        free(content);
                        content = h3_json_string(json);
                        if (!content) {
                            free(key);
                            return 0;
                        }
                    } else if (!strcmp(key, "single_word") ||
                               !strcmp(key, "lstrip") ||
                               !strcmp(key, "rstrip") ||
                               !strcmp(key, "normalized")) {
                        int flag;
                        if (!h3_json_bool(json, &flag)) {
                            free(key);
                            free(content);
                            return 0;
                        }
                        if (flag) {
                            free(key);
                            free(content);
                            return h3_json_fail(json,
                                "unsupported added-token policy");
                        }
                    } else {
                        if (!h3_json_skip(json)) {
                            free(key);
                            free(content);
                            return 0;
                        }
                    }
                    free(key);
                    if (h3_json_take(json, ',')) continue;
                    break;
                }
            }
            if (!h3_json_expect(json, '}')) {
                free(content);
                return 0;
            }
            if (!have_id || !content || id > UINT32_MAX) {
                free(content);
                return h3_json_fail(json,
                                    "unexpected tokenizer specification");
            }
            if ((uint32_t)id > *maximum_id) *maximum_id = (uint32_t)id;
            /* inverse_added is rebuilt from the map once parsing finishes. */
            if (!h3_map_find(&tokenizer->added, content)) {
                char **grown = realloc(tokenizer->alternatives,
                    (tokenizer->alternative_count + 1) * sizeof(*grown));
                if (!grown) {
                    free(content);
                    return h3_json_fail(json,
                                        "out of memory loading tokenizer");
                }
                tokenizer->alternatives = grown;
                tokenizer->alternatives[tokenizer->alternative_count++] =
                    content;
            }
            if (!h3_map_put_u32(&tokenizer->added, content, (uint32_t)id))
                return h3_json_fail(json, "out of memory loading tokenizer");
            if (h3_json_take(json, ',')) continue;
            break;
        }
    }
    return h3_json_expect(json, ']');
}

static int h3_parse_model(h3_json *json, h3_tokenizer *tokenizer,
                          uint32_t *maximum_id, int *valid) {
    int type_ok = 0;
    int have_unk = 0;
    int have_vocab = 0;
    int have_merges = 0;
    if (!h3_json_expect(json, '{')) return 0;
    if (!h3_json_peek(json, '}')) {
        for (;;) {
            char *key = h3_json_string(json);
            if (!key) return 0;
            if (!h3_json_expect(json, ':')) {
                free(key);
                return 0;
            }
            if (!strcmp(key, "type")) {
                char *type = h3_json_string(json);
                if (!type) {
                    free(key);
                    return 0;
                }
                type_ok = !strcmp(type, "BPE");
                free(type);
            } else if (!strcmp(key, "unk_token")) {
                if (!h3_json_null(json)) {
                    free(key);
                    return 0;
                }
                have_unk = 1;
            } else if (!strcmp(key, "vocab")) {
                if (!h3_parse_vocab(json, tokenizer, maximum_id)) {
                    free(key);
                    return 0;
                }
                have_vocab = 1;
            } else if (!strcmp(key, "merges")) {
                if (!h3_parse_merges(json, tokenizer)) {
                    free(key);
                    return 0;
                }
                have_merges = 1;
            } else {
                if (!h3_json_skip(json)) {
                    free(key);
                    return 0;
                }
            }
            free(key);
            if (h3_json_take(json, ',')) continue;
            break;
        }
    }
    if (!h3_json_expect(json, '}')) return 0;
    *valid = type_ok && have_unk && have_vocab && have_merges;
    return 1;
}

static int h3_parse_normalizer(h3_json *json, int *valid) {
    int type_seen = 0;
    int type_ok = 0;
    if (!h3_json_expect(json, '{')) return 0;
    if (!h3_json_peek(json, '}')) {
        for (;;) {
            char *key = h3_json_string(json);
            if (!key) return 0;
            if (!h3_json_expect(json, ':')) {
                free(key);
                return 0;
            }
            if (!strcmp(key, "type")) {
                char *type = h3_json_string(json);
                if (!type) {
                    free(key);
                    return 0;
                }
                type_seen = 1;
                type_ok = !strcmp(type, "NFC");
                free(type);
            } else {
                if (!h3_json_skip(json)) {
                    free(key);
                    return 0;
                }
            }
            free(key);
            if (h3_json_take(json, ',')) continue;
            break;
        }
    }
    if (!h3_json_expect(json, '}')) return 0;
    *valid = type_seen && type_ok;
    return 1;
}

static int h3_alternative_compare(const void *left, const void *right) {
    const char *a = *(char *const *)left;
    const char *b = *(char *const *)right;
    size_t a_length = strlen(a);
    size_t b_length = strlen(b);
    if (a_length != b_length) return a_length > b_length ? -1 : 1;
    return strcmp(a, b);
}

static char *h3_slurp(const char *path, size_t *out_size) {
    FILE *file = fopen(path, "rb");
    if (!file) return NULL;
    if (fseek(file, 0, SEEK_END)) {
        fclose(file);
        return NULL;
    }
    long size = ftell(file);
    if (size < 0 || fseek(file, 0, SEEK_SET)) {
        fclose(file);
        return NULL;
    }
    char *data = malloc((size_t)size + 1);
    if (!data) {
        fclose(file);
        return NULL;
    }
    if (size && fread(data, 1, (size_t)size, file) != (size_t)size) {
        free(data);
        fclose(file);
        return NULL;
    }
    fclose(file);
    data[size] = '\0';
    if (out_size) *out_size = (size_t)size;
    return data;
}

h3_tokenizer *h3_tokenizer_load(const char *path, char *error,
                                size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!path) {
        h3_tok_error(error, error_size, "tokenizer path is required");
        return NULL;
    }
    size_t size = 0;
    char *data = h3_slurp(path, &size);
    if (!data) {
        char message[512];
        snprintf(message, sizeof(message), "cannot read tokenizer: %s", path);
        h3_tok_error(error, error_size, message);
        return NULL;
    }
    h3_tokenizer *tokenizer = calloc(1, sizeof(*tokenizer));
    if (!tokenizer) {
        free(data);
        h3_tok_error(error, error_size, "out of memory loading tokenizer");
        return NULL;
    }
    h3_json json = {data, data + size, {0}};
    uint32_t maximum_id = 0;
    int have_model = 0;
    int model_ok = 0;
    int have_normalizer = 0;
    int normalizer_ok = 0;
    if (h3_utf8_valid(data, size) && h3_json_expect(&json, '{')) {
        if (!h3_json_peek(&json, '}')) {
            for (;;) {
                char *key = h3_json_string(&json);
                if (!key) break;
                if (!h3_json_expect(&json, ':')) {
                    free(key);
                    break;
                }
                if (!strcmp(key, "model")) {
                    have_model = 1;
                    if (!h3_parse_model(&json, tokenizer, &maximum_id,
                                        &model_ok)) {
                        free(key);
                        break;
                    }
                } else if (!strcmp(key, "normalizer")) {
                    have_normalizer = 1;
                    if (!h3_parse_normalizer(&json, &normalizer_ok)) {
                        free(key);
                        break;
                    }
                } else if (!strcmp(key, "added_tokens")) {
                    if (!h3_parse_added(&json, tokenizer, &maximum_id)) {
                        free(key);
                        break;
                    }
                } else {
                    if (!h3_json_skip(&json)) {
                        free(key);
                        break;
                    }
                }
                free(key);
                if (h3_json_take(&json, ',')) continue;
                break;
            }
        }
        if (!json.error[0] && h3_json_expect(&json, '}')) {
            h3_json_ws(&json);
            if (json.at != json.end) h3_json_syntax(&json);
        }
    } else if (!json.error[0]) {
        h3_json_syntax(&json);
    }
    if (!json.error[0] && (!have_model || !model_ok || !have_normalizer ||
                           !normalizer_ok))
        h3_json_fail(&json, "unexpected tokenizer specification");
    if (json.error[0]) {
        h3_tok_error(error, error_size, json.error);
        h3_tokenizer_free(tokenizer);
        free(data);
        return NULL;
    }
    tokenizer->inverse_count = (size_t)maximum_id + 1;
    tokenizer->inverse_vocab =
        calloc(tokenizer->inverse_count, sizeof(*tokenizer->inverse_vocab));
    tokenizer->inverse_added =
        calloc(tokenizer->inverse_count, sizeof(*tokenizer->inverse_added));
    if (!tokenizer->inverse_vocab || !tokenizer->inverse_added) {
        h3_tok_error(error, error_size, "out of memory loading tokenizer");
        h3_tokenizer_free(tokenizer);
        free(data);
        return NULL;
    }
    for (size_t index = 0; index < tokenizer->vocab.capacity; index++) {
        h3_slot *slot = &tokenizer->vocab.slots[index];
        if (slot->key) tokenizer->inverse_vocab[slot->value] = slot->key;
    }
    for (size_t index = 0; index < tokenizer->added.capacity; index++) {
        h3_slot *slot = &tokenizer->added.slots[index];
        if (slot->key) tokenizer->inverse_added[slot->value] = slot->key;
    }
    if (tokenizer->alternative_count > 1)
        qsort(tokenizer->alternatives, tokenizer->alternative_count,
              sizeof(*tokenizer->alternatives), h3_alternative_compare);

    for (size_t index = 0; index < H3_BYTE_DECODER_SIZE; index++)
        tokenizer->byte_decoder[index] = -1;
    {
        unsigned extra = 0;
        for (unsigned byte = 0; byte < 256; byte++) {
            int visible = (byte >= '!' && byte <= '~') ||
                          (byte >= 0xa1 && byte <= 0xac) ||
                          (byte >= 0xae && byte <= 0xff);
            uint32_t codepoint = visible ? byte : 256u + extra++;
            tokenizer->byte_encoder[byte] = codepoint;
            tokenizer->byte_decoder[codepoint] = (int16_t)byte;
        }
    }
    free(data);
    return tokenizer;
}

void h3_tokenizer_free(h3_tokenizer *tokenizer) {
    if (!tokenizer) return;
    h3_map_clear(&tokenizer->vocab);
    h3_map_clear(&tokenizer->merges);
    h3_map_clear(&tokenizer->added);
    h3_map_clear(&tokenizer->cache);
    free(tokenizer->alternatives);
    free(tokenizer->inverse_vocab);
    free(tokenizer->inverse_added);
    free(tokenizer);
}

/* ---------- pre-tokenization ---------- */

typedef struct {
    uint32_t value;
    size_t location; /* UTF-8 byte offset */
    size_t length;   /* UTF-8 byte length */
} h3_codepoint;

static h3_codepoint *h3_codepoints(const char *text, size_t length,
                                   size_t *count) {
    h3_codepoint *points = calloc(length ? length : 1, sizeof(*points));
    if (!points) return NULL;
    size_t used = 0;
    size_t index = 0;
    while (index < length) {
        size_t width;
        uint32_t value = h3_utf8_decode(text + index, length - index, &width);
        points[used++] = (h3_codepoint){value, index, width};
        index += width;
    }
    *count = used;
    return points;
}

static size_t h3_contraction(const h3_codepoint *points, size_t count,
                             size_t index) {
    static const char *values[] = {"'s", "'t", "'re", "'ve",
                                   "'m", "'ll", "'d"};
    if (points[index].value != '\'') return 0;
    for (size_t item = 0; item < sizeof(values) / sizeof(values[0]); item++) {
        size_t length = strlen(values[item]);
        if (index + length > count) continue;
        int matches = 1;
        for (size_t offset = 1; offset < length; offset++) {
            uint32_t got = points[index + offset].value;
            if (got >= 'A' && got <= 'Z') got += 'a' - 'A';
            if (got != (unsigned char)values[item][offset]) matches = 0;
        }
        if (matches) return length;
    }
    return 0;
}

typedef struct {
    size_t start; /* UTF-8 byte offsets into the normalized text */
    size_t end;
} h3_span;

static int h3_span_push(h3_span **spans, size_t *used, size_t *capacity,
                        const h3_codepoint *points, size_t start,
                        size_t stop) {
    if (*used == *capacity) {
        size_t next = *capacity ? *capacity * 2 : 16;
        h3_span *grown = realloc(*spans, next * sizeof(*grown));
        if (!grown) return 0;
        *spans = grown;
        *capacity = next;
    }
    (*spans)[*used].start = points[start].location;
    (*spans)[*used].end = points[stop - 1].location + points[stop - 1].length;
    (*used)++;
    return 1;
}

/* Exact port of h3_pretokenize in h3_tokenizer.m. */
static int h3_pretokenize(const char *text, size_t length, h3_span **out,
                          size_t *out_count) {
    size_t count = 0;
    h3_codepoint *points = h3_codepoints(text, length, &count);
    if (!points) return 0;
    h3_span *spans = NULL;
    size_t used = 0;
    size_t capacity = 0;
    size_t index = 0;
    while (index < count) {
        size_t contraction = h3_contraction(points, count, index);
        if (contraction) {
            if (!h3_span_push(&spans, &used, &capacity, points, index,
                              index + contraction))
                goto fail;
            index += contraction;
            continue;
        }
        uint32_t value = points[index].value;
        ptrdiff_t letter_start = (ptrdiff_t)index;
        if (h3_letter(value)) {
            /* Already at the first letter. */
        } else if (value != '\r' && value != '\n' && !h3_number(value) &&
                   index + 1 < count && h3_letter(points[index + 1].value)) {
            letter_start++;
        } else {
            letter_start = -1;
        }
        if (letter_start >= 0) {
            size_t stop = (size_t)letter_start;
            while (stop < count && h3_letter(points[stop].value)) stop++;
            if (!h3_span_push(&spans, &used, &capacity, points, index, stop))
                goto fail;
            index = stop;
            continue;
        }
        if (h3_number(value)) {
            if (!h3_span_push(&spans, &used, &capacity, points, index,
                              index + 1))
                goto fail;
            index++;
            continue;
        }
        size_t punct_start = index +
            (value == ' ' && index + 1 < count &&
             !h3_space(points[index + 1].value) &&
             !h3_letter(points[index + 1].value) &&
             !h3_number(points[index + 1].value));
        size_t stop = punct_start;
        while (stop < count && !h3_space(points[stop].value) &&
               !h3_letter(points[stop].value) &&
               !h3_number(points[stop].value)) stop++;
        if (stop > punct_start) {
            while (stop < count &&
                   (points[stop].value == '\r' || points[stop].value == '\n'))
                stop++;
            if (!h3_span_push(&spans, &used, &capacity, points, index, stop))
                goto fail;
            index = stop;
            continue;
        }
        if (h3_space(value)) {
            size_t whitespace_end = index + 1;
            while (whitespace_end < count &&
                   h3_space(points[whitespace_end].value)) whitespace_end++;
            ptrdiff_t newline_end = -1;
            for (size_t cursor = index; cursor < whitespace_end; cursor++) {
                if (points[cursor].value == '\r' ||
                    points[cursor].value == '\n')
                    newline_end = (ptrdiff_t)cursor + 1;
            }
            size_t piece_end;
            if (newline_end >= 0) piece_end = (size_t)newline_end;
            else if (whitespace_end == count) piece_end = whitespace_end;
            else if (whitespace_end - index > 1) piece_end = whitespace_end - 1;
            else piece_end = index + 1;
            if (!h3_span_push(&spans, &used, &capacity, points, index,
                              piece_end))
                goto fail;
            index = piece_end;
            continue;
        }
        goto fail; /* Unreachable: every codepoint has a class above. */
    }
    free(points);
    *out = spans;
    *out_count = used;
    return 1;
fail:
    free(points);
    free(spans);
    return 0;
}

/* ---------- BPE ---------- */

typedef struct {
    uint32_t *ids;
    size_t used;
    size_t capacity;
} h3_idvec;

static int h3_idvec_push(h3_idvec *vector, uint32_t id) {
    if (vector->used == vector->capacity) {
        size_t next = vector->capacity ? vector->capacity * 2 : 16;
        uint32_t *grown = realloc(vector->ids, next * sizeof(*grown));
        if (!grown) return 0;
        vector->ids = grown;
        vector->capacity = next;
    }
    vector->ids[vector->used++] = id;
    return 1;
}

static void h3_symbols_free(char **symbols, size_t count) {
    for (size_t index = 0; index < count; index++) free(symbols[index]);
    free(symbols);
}

static int h3_bpe(h3_tokenizer *tokenizer, const char *piece,
                  size_t piece_length, h3_idvec *output, char *failure,
                  size_t failure_size) {
    /* Byte-encode the piece (GPT-2 bytes_to_unicode). */
    char *encoded = malloc(piece_length * 2 + 1);
    if (!encoded) {
        snprintf(failure, failure_size, "out of memory encoding prompt");
        return 0;
    }
    size_t encoded_length = 0;
    for (size_t index = 0; index < piece_length; index++) {
        char unit[4];
        size_t width = h3_utf8_encode(
            tokenizer->byte_encoder[(unsigned char)piece[index]], unit);
        memcpy(encoded + encoded_length, unit, width);
        encoded_length += width;
    }
    encoded[encoded_length] = '\0';

    const h3_slot *cached = h3_map_find(&tokenizer->cache, encoded);
    if (cached) {
        for (size_t index = 0; index < cached->count; index++) {
            if (!h3_idvec_push(output, cached->ids[index])) {
                free(encoded);
                snprintf(failure, failure_size, "out of memory encoding prompt");
                return 0;
            }
        }
        free(encoded);
        return 1;
    }

    /* Split the encoded piece into single-codepoint symbols. */
    char **symbols = malloc(encoded_length * sizeof(*symbols));
    if (!symbols) {
        free(encoded);
        snprintf(failure, failure_size, "out of memory encoding prompt");
        return 0;
    }
    size_t count = 0;
    size_t offset = 0;
    while (offset < encoded_length) {
        size_t width;
        h3_utf8_decode(encoded + offset, encoded_length - offset, &width);
        symbols[count] = h3_strndup(encoded + offset, width);
        if (!symbols[count]) {
            h3_symbols_free(symbols, count);
            free(encoded);
            snprintf(failure, failure_size, "out of memory encoding prompt");
            return 0;
        }
        count++;
        offset += width;
    }

    /* Greedy lowest-rank merges. */
    while (count > 1) {
        uint32_t best_rank = 0;
        size_t best = 0;
        int have_best = 0;
        for (size_t index = 0; index + 1 < count; index++) {
            char *key = h3_pair_key(symbols[index], strlen(symbols[index]),
                                    symbols[index + 1],
                                    strlen(symbols[index + 1]));
            uint32_t rank;
            int found;
            if (!key) {
                h3_symbols_free(symbols, count);
                free(encoded);
                snprintf(failure, failure_size, "out of memory encoding prompt");
                return 0;
            }
            found = h3_map_get_u32(&tokenizer->merges, key, &rank);
            free(key);
            if (found && (!have_best || rank < best_rank)) {
                best_rank = rank;
                best = index;
                have_best = 1;
            }
        }
        if (!have_best) break;
        char *left = h3_strdup(symbols[best]);
        char *right = h3_strdup(symbols[best + 1]);
        if (!left || !right) {
            free(left);
            free(right);
            h3_symbols_free(symbols, count);
            free(encoded);
            snprintf(failure, failure_size, "out of memory encoding prompt");
            return 0;
        }
        size_t written = 0;
        for (size_t index = 0; index < count;) {
            if (index + 1 < count && !strcmp(symbols[index], left) &&
                !strcmp(symbols[index + 1], right)) {
                size_t left_length = strlen(symbols[index]);
                size_t right_length = strlen(symbols[index + 1]);
                char *merged = malloc(left_length + right_length + 1);
                if (!merged) {
                    free(left);
                    free(right);
                    h3_symbols_free(symbols, count);
                    free(encoded);
                    snprintf(failure, failure_size,
                             "out of memory encoding prompt");
                    return 0;
                }
                memcpy(merged, symbols[index], left_length);
                memcpy(merged + left_length, symbols[index + 1], right_length);
                merged[left_length + right_length] = '\0';
                free(symbols[index]);
                free(symbols[index + 1]);
                symbols[written++] = merged;
                index += 2;
            } else {
                symbols[written++] = symbols[index++];
            }
        }
        count = written;
        free(left);
        free(right);
    }

    h3_idvec ids = {NULL, 0, 0};
    for (size_t index = 0; index < count; index++) {
        uint32_t id;
        if (!h3_map_get_u32(&tokenizer->vocab, symbols[index], &id)) {
            snprintf(failure, failure_size,
                     "BPE symbol is absent from vocabulary: %s",
                     symbols[index]);
            free(ids.ids);
            h3_symbols_free(symbols, count);
            free(encoded);
            return 0;
        }
        if (!h3_idvec_push(&ids, id)) {
            free(ids.ids);
            h3_symbols_free(symbols, count);
            free(encoded);
            snprintf(failure, failure_size, "out of memory encoding prompt");
            return 0;
        }
    }
    h3_symbols_free(symbols, count);
    if (!h3_map_put_ids(&tokenizer->cache, encoded, ids.ids, ids.used)) {
        free(ids.ids);
        snprintf(failure, failure_size, "out of memory encoding prompt");
        return 0;
    }
    for (size_t index = 0; index < ids.used; index++) {
        if (!h3_idvec_push(output, ids.ids[index])) {
            free(ids.ids);
            snprintf(failure, failure_size, "out of memory encoding prompt");
            return 0;
        }
    }
    free(ids.ids);
    return 1;
}

static int h3_encode_plain(h3_tokenizer *tokenizer, const char *text,
                           size_t length, h3_idvec *output, char *failure,
                           size_t failure_size) {
    size_t normalized_length = 0;
    char *normalized = h3_nfc(text, length, &normalized_length);
    if (!normalized) {
        snprintf(failure, failure_size, "unable to pre-tokenize input");
        return 0;
    }
    h3_span *pieces = NULL;
    size_t piece_count = 0;
    if (!h3_pretokenize(normalized, normalized_length, &pieces,
                        &piece_count)) {
        free(normalized);
        snprintf(failure, failure_size, "unable to pre-tokenize input");
        return 0;
    }
    for (size_t index = 0; index < piece_count; index++) {
        if (!h3_bpe(tokenizer, normalized + pieces[index].start,
                    pieces[index].end - pieces[index].start, output, failure,
                    failure_size)) {
            free(pieces);
            free(normalized);
            return 0;
        }
    }
    free(pieces);
    free(normalized);
    return 1;
}

/* ---------- public encode / decode ---------- */

/* First occurrence of needle at or after haystack start, or NULL. */
static const char *h3_find(const char *haystack, size_t haystack_length,
                           const char *needle, size_t needle_length) {
    if (needle_length > haystack_length) return NULL;
    for (size_t index = 0; index + needle_length <= haystack_length;
         index++) {
        if (!memcmp(haystack + index, needle, needle_length))
            return haystack + index;
    }
    return NULL;
}

int h3_tokenizer_encode(const h3_tokenizer *opaque, const char *utf8,
                        int pad_empty, uint32_t **ids, size_t *count,
                        char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!opaque || !utf8 || !ids || !count) return 0;
    *ids = NULL;
    *count = 0;
    size_t length = strlen(utf8);
    if (!h3_utf8_valid(utf8, length)) {
        h3_tok_error(error, error_size, "prompt is not valid UTF-8");
        return 0;
    }
    h3_tokenizer *tokenizer = (h3_tokenizer *)opaque;
    h3_idvec output = {NULL, 0, 0};
    char failure[512];
    failure[0] = '\0';
    size_t start = 0;
    while (start < length) {
        /* Leftmost match wins; ties go to the longest alternative. */
        size_t match_location = length;
        size_t match_length = 0;
        uint32_t match_id = 0;
        int found = 0;
        for (size_t item = 0; item < tokenizer->alternative_count; item++) {
            const char *candidate = tokenizer->alternatives[item];
            size_t candidate_length = strlen(candidate);
            const char *hit;
            if (!candidate_length) continue;
            hit = h3_find(utf8 + start, length - start, candidate,
                          candidate_length);
            if (!hit) continue;
            size_t location = (size_t)(hit - utf8);
            if (!found || location < match_location ||
                (location == match_location &&
                 candidate_length > match_length)) {
                match_location = location;
                match_length = candidate_length;
                {
                    const h3_slot *slot =
                        h3_map_find(&tokenizer->added, candidate);
                    match_id = slot ? slot->value : 0;
                }
                found = 1;
            }
        }
        if (!found) break;
        if (match_location > start &&
            !h3_encode_plain(tokenizer, utf8 + start, match_location - start,
                             &output, failure, sizeof(failure)))
            goto fail;
        if (!h3_idvec_push(&output, match_id)) {
            snprintf(failure, sizeof(failure), "out of memory encoding prompt");
            goto fail;
        }
        start = match_location + match_length;
    }
    if (start < length &&
        !h3_encode_plain(tokenizer, utf8 + start, length - start, &output,
                         failure, sizeof(failure)))
        goto fail;
    if (output.used == 0 && pad_empty &&
        !h3_idvec_push(&output, H3_PAD_TOKEN_ID)) {
        snprintf(failure, sizeof(failure), "out of memory encoding prompt");
        goto fail;
    }
    if (output.used) {
        *ids = output.ids;
    } else {
        free(output.ids);
    }
    *count = output.used;
    return 1;
fail:
    h3_tok_error(error, error_size,
                 failure[0] ? failure : "tokenizer failure");
    free(output.ids);
    return 0;
}

void h3_tokenizer_ids_free(uint32_t *ids) {
    free(ids);
}

typedef struct {
    char *bytes;
    size_t used;
    size_t capacity;
} h3_bytevec;

static int h3_bytevec_push(h3_bytevec *vector, const char *bytes,
                           size_t count) {
    if (vector->used + count > vector->capacity) {
        size_t next = vector->capacity ? vector->capacity * 2 : 64;
        while (vector->used + count > next) next *= 2;
        char *grown = realloc(vector->bytes, next);
        if (!grown) return 0;
        vector->bytes = grown;
        vector->capacity = next;
    }
    memcpy(vector->bytes + vector->used, bytes, count);
    vector->used += count;
    return 1;
}

/* Append pending bytes to result; invalid UTF-8 becomes U+FFFD, matching
 * NSString's initWithData:encoding: failure path in h3_tokenizer.m. */
static int h3_decode_flush(h3_bytevec *result, h3_bytevec *pending) {
    if (!pending->used) return 1;
    int ok = h3_utf8_valid(pending->bytes, pending->used)
                 ? h3_bytevec_push(result, pending->bytes, pending->used)
                 : h3_bytevec_push(result, "\xef\xbf\xbd", 3);
    pending->used = 0;
    return ok;
}

char *h3_tokenizer_decode(const h3_tokenizer *opaque,
                          const uint32_t *ids, size_t count,
                          char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!opaque || (!ids && count)) return NULL;
    const h3_tokenizer *tokenizer = opaque;
    h3_bytevec result = {NULL, 0, 0};
    h3_bytevec pending = {NULL, 0, 0};
    for (size_t index = 0; index < count; index++) {
        uint32_t identifier = ids[index];
        if ((size_t)identifier >= tokenizer->inverse_count) {
            h3_tok_error(error, error_size, "token ID is out of range");
            goto fail;
        }
        const char *added = tokenizer->inverse_added[identifier];
        if (added) {
            if (!h3_decode_flush(&result, &pending) ||
                !h3_bytevec_push(&result, added, strlen(added))) {
                h3_tok_error(error, error_size, "out of memory decoding tokens");
                goto fail;
            }
            continue;
        }
        const char *symbol = tokenizer->inverse_vocab[identifier];
        if (!symbol) {
            h3_tok_error(error, error_size, "unknown token ID");
            goto fail;
        }
        size_t symbol_length = strlen(symbol);
        size_t offset = 0;
        while (offset < symbol_length) {
            size_t width;
            uint32_t codepoint = h3_utf8_decode(symbol + offset,
                                                symbol_length - offset,
                                                &width);
            offset += width;
            if (codepoint >= H3_BYTE_DECODER_SIZE ||
                tokenizer->byte_decoder[codepoint] < 0) {
                h3_tok_error(error, error_size, "invalid byte-level token");
                goto fail;
            }
            {
                char byte = (char)tokenizer->byte_decoder[codepoint];
                if (!h3_bytevec_push(&pending, &byte, 1)) {
                    h3_tok_error(error, error_size,
                                 "out of memory decoding tokens");
                    goto fail;
                }
            }
        }
    }
    if (!h3_decode_flush(&result, &pending) ||
        !h3_bytevec_push(&result, "", 1)) {
        h3_tok_error(error, error_size, "out of memory decoding tokens");
        goto fail;
    }
    free(pending.bytes);
    result.bytes[result.used - 1] = '\0';
    return result.bytes;
fail:
    free(result.bytes);
    free(pending.bytes);
    return NULL;
}
