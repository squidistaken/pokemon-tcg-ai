// SPDX-FileCopyrightText: © Pokémon/Nintendo/Creatures/GAME FREAK TM, ®, and character names are trademarks of Nintendo.
// SPDX-License-Identifier: LicenseRef-PTCG-ABC-Competition-Use-Only
// Part of the Pokémon TCG AI Battle Challenge. Provided for Competition use only;
// the full license is in the LICENSES/ folder and incorporates the Competition Rules.
// Competition Rules: https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/rules

#pragma once

#include "Core.h"

class JsonBuilder {
public:
	std::u8string buf;

	void clear() {
		buf.clear();
	}

	void appendDoubleQuote(const char* str) {
		append(u8'"');
		buf += (const char8_t*)str;
		append(u8'"');
	}

	void appendDoubleQuote(const char8_t* str) {
		append(u8'"');
		buf += str;
		append(u8'"');
	}

	void appendDoubleQuote(const std::u8string& str) {
		append(u8'"');
		buf += str;
		append(u8'"');
	}

	template<int Size>
	void appendStr(const char(&ss)[Size]) {
		std::u8string_view v{ (const char8_t*)ss, Size - 1 };
		buf += v;
	}

	void append(bool c) {
		if (c) {
			appendStr("true");
		} else {
			appendStr("false");
		}
	}

	void append(char8_t c) {
		buf += c;
	}

	void append(char c) {
		buf += (char8_t)c;
	}

	void append(int num) {
		if (num < 0) {
			num = -num;
			append(u8'-');
		}
		if (num <= 9) {
			append((char8_t)(u8'0' + num));
			return;
		}

		unsigned n = num;
		char8_t c[16];
		int digit = 0;
		while (n > 0) {
			c[digit] = (char8_t)(u8'0' + n % 10);
			digit++;
			n = n / 10;
		}
		for (int i = digit - 1; i >= 0; i--) {
			buf += c[i];
		}
	}

	void append(signed char num) {
		append((int)num);
	}

	void append(unsigned char num) {
		append((int)num);
	}

	void append(size_t num) {
		append((int)num);
	}

	void appendNull() {
		appendStr("null");
	}

	void comma(int index) {
		if (index != 0) {
			append(u8',');
		}
	}


	template<int Size>
	void appendKey(const char(&key)[Size]) {
		append(u8'"');
		appendStr(key);
		append(u8'"');
		append(u8':');
	}

	template<int Size>
	void appendCommaKey(const char(&key)[Size]) {
		append(u8',');
		appendKey(key);
	}

	template<int Size>
	void appendKeyValue(const char(&key)[Size], int val) {
		appendKey(key);
		append(val);
	}
	template<int Size>
	void appendKeyValue(const char(&key)[Size], bool val) {
		appendKey(key);
		append(val);
	}
	template<int Size>
	void appendKeyValue(const char(&key)[Size], const std::u8string& val) {
		appendKey(key);
		appendDoubleQuote(val);
	}

	template<int Size>
	void appendCommaKeyValue(const char(&key)[Size], int val) {
		append(u8',');
		appendKeyValue(key, val);
	}
	template<int Size>
	void appendCommaKeyValue(const char(&key)[Size], bool val) {
		append(u8',');
		appendKeyValue(key, val);
	}
	template<int Size>
	void appendCommaKeyValue(const char(&key)[Size], const std::u8string& val) {
		append(u8',');
		appendKeyValue(key, val);
	}

	// 負の数ならnull
	template<int Size>
	void appendCommaKeyValueOrNull(const char(&key)[Size], int val) {
		append(u8',');
		appendKey(key);
		if (val >= 0) {
			append(val);
		} else {
			appendNull();
		}
	}

	// 空文字列ならnull
	template<int Size>
	void appendCommaKeyValueOrNull(const char(&key)[Size], const std::u8string& val) {
		append(u8',');
		appendKey(key);
		if (val.empty()) {
			appendNull();
		} else {
			appendDoubleQuote(val);
		}
	}
};
