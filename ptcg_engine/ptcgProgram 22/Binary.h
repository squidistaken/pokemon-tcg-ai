// SPDX-FileCopyrightText: © Pokémon/Nintendo/Creatures/GAME FREAK TM, ®, and character names are trademarks of Nintendo.
// SPDX-License-Identifier: LicenseRef-PTCG-ABC-Competition-Use-Only
// Part of the Pokémon TCG AI Battle Challenge. Provided for Competition use only;
// the full license is in the LICENSES/ folder and incorporates the Competition Rules.
// Competition Rules: https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/rules

#pragma once

#include "Core.h"

static inline const std::string base64String = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

class BinaryWriter {
public:

	std::vector<uint8_t> buf;
	std::vector<char> base64;
	int countA = 0;
	int countSrc = 0;

	void set(const void* p, size_t numBytes) {
		if (numBytes <= 0) {
			return;
		}
		size_t first = buf.size();
		buf.resize(first + numBytes);
		const uint8_t* ptr = (const uint8_t*)p;
		std::copy(ptr, ptr + numBytes, &buf.at(first));
	}

	void set(const void* pFirst, const void* pLast) {
		set(pFirst, (const uint8_t*)pLast - (const uint8_t*)pFirst);
	}

	template<typename T>
	void set(const std::vector<T>& list) {
		set((int)list.size());
		set(list.data(), sizeof(T) * list.size());
	}

	void set(int num) {
		set(&num, sizeof(int));
	}

	void clear() {
		buf.clear();
	}

	int size() const {
		return (int)buf.size();
	}

	void addA() {
		if (countA > 0) {
			int c = countA;
			if (countA < 64) {
				base64.push_back('A');
				base64.push_back(base64String[c]);
			} else if (countA < 64 * 64) {
				base64.push_back('-');
				base64.push_back(base64String[c % 64]);
				base64.push_back(base64String[c / 64]);
			} else {
				base64.push_back('*');
				int c2 = c / 64;
				base64.push_back(base64String[c % 64]);
				base64.push_back(base64String[c2 % 64]);
				base64.push_back(base64String[c2 / 64]);
			}
			countSrc += countA;
			countA = 0;
		}
	}

	void addChar(char c) {
		if (c == 'A') {
			countA++;
		} else {
			addA();
			base64.push_back(c);
			countSrc++;
		}
	}

	void toBase64() {
		countA = 0;
		base64.clear();
		base64.reserve(buf.size() * 4 / 3 + 1);
		int val = 0;
		int sf = -6;
		for (unsigned char c : buf) {
			val = (val << 8) + c;
			sf += 8;
			while (sf >= 0) {
				addChar(base64String[(val >> sf) & 0x3F]);
				sf -= 6;
			}
		}
		if (sf > -6) {
			addChar(base64String[((val << 8) >> (sf + 8)) & 0x3F]);
		}
		addA();
		while (countSrc % 4) {
			base64.push_back('=');
			countSrc++;
		}
	}
};

class BinaryReader {
public:

	std::vector<uint8_t> buf;
	std::vector<char> base64;
	std::vector<char> base64Dest;
	size_t pos = 0;

	void set(void* p, size_t numBytes) {
		auto start = buf.begin() + pos;
		std::copy(start, start + numBytes, (uint8_t*)p);
		pos += numBytes;
	}

	void set(void* pFirst, void* pLast) {
		set(pFirst, (uint8_t*)pLast - (uint8_t*)pFirst);
	}

	template<typename T>
	void set(std::vector<T>& list) {
		list.resize(readInt());
		set(list.data(), sizeof(T) * list.size());
	}

	int readInt() {
		int v = buf[pos] + (buf[pos + 1] << 8) + (buf[pos + 2] << 16) + (buf[pos + 3] << 24);
		pos += 4;
		return v;
	}

	void fromBase64() {
		base64Dest.clear();
		int last = (int)base64.size();
		for (int i = 0; i < last; i++) {
			char c = base64[i];
			int countA = 0;
			if (c == 'A') {
				i++;
				countA = toValue(base64[i]);
			} else if (c == '-') {
				i++;
				countA += toValue(base64[i]);
				i++;
				countA += 64 * toValue(base64[i]);
			} else if (c == '*') {
				i++;
				countA += toValue(base64[i]);
				i++;
				countA += 64 * toValue(base64[i]);
				i++;
				countA += 64 * 64 * toValue(base64[i]);
			} else {
				base64Dest.push_back(c);
				continue;
			}
			for (int j = 0; j < countA; j++) {
				base64Dest.push_back('A');
			}
		}

		int padding = 0;
		int inputCount = (int)base64Dest.size();

		if (base64Dest[inputCount - 1] == '=') {
			padding++;
		}
		if (base64Dest[inputCount - 2] == '=') {
			padding++;
		}

		int outputCount = (inputCount / 4) * 3 - padding;
		pos = 0;
		buf.resize(outputCount);

		int j = 0;
		for (int i = 0; i < inputCount; i += 4) {
			int val = (toValue(base64Dest[i]) << 18) +
				(toValue(base64Dest[i + 1]) << 12) +
				(toValue(base64Dest[i + 2]) << 6) +
				toValue(base64Dest[i + 3]);

			if (j < outputCount) {
				buf[j++] = (val >> 16) & 0xFF;
			}
			if (j < outputCount) {
				buf[j++] = (val >> 8) & 0xFF;
			}
			if (j < outputCount) {
				buf[j++] = val & 0xFF;
			}
		}
	}

	int toValue(char c) {
		if ('A' <= c && c <= 'Z') {
			return c - 'A';
		}
		if ('a' <= c && c <= 'z') {
			return c - 'a' + 26;
		}
		if ('0' <= c && c <= '9') {
			return c - '0' + 52;
		}
		if (c == '+') {
			return 62;
		}
		if (c == '/') {
			return 63;
		}
		return 0;
	}
};
