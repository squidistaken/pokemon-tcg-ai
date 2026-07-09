// SPDX-FileCopyrightText: © Pokémon/Nintendo/Creatures/GAME FREAK TM, ®, and character names are trademarks of Nintendo.
// SPDX-License-Identifier: LicenseRef-PTCG-ABC-Competition-Use-Only
// Part of the Pokémon TCG AI Battle Challenge. Provided for Competition use only;
// the full license is in the LICENSES/ folder and incorporates the Competition Rules.
// Competition Rules: https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/rules

#pragma once

#include "Core.h"

// 最大容量を指定するリスト
template<typename SizeT, typename T, SizeT Capacity>
class FixedListBase {
public:

	FixedListBase() noexcept {
		cnt = 0;
	}
	FixedListBase(const FixedListBase<SizeT, T, Capacity>& src) noexcept {
		operator=(src);
	}

	void operator=(const FixedListBase<SizeT, T, Capacity>& src) noexcept {
		cnt = src.cnt;
		std::copy(src.begin(), src.end(), begin());
	}

	void push_back(const T& t) {
		checkFull();
		buffer[cnt++] = t;
	}

	T& emplace_back() {
		checkFull();
		return buffer[cnt++];
	}

	void push_front(const T& t) {
		cnt++;
		for (int i = cnt - 1; i >= 1; i--) {
			buffer[i] = buffer[i - 1];
		}
		buffer[0] = t;
	}

	void pop_back() {
		checkEmpty();
		cnt--;
	}

	T pop() {
		checkEmpty();
		T t = buffer[cnt - 1];
		cnt--;
		return t;
	}

	int size() const {
		return cnt;
	}

	[[nodiscard]]
	bool empty() const {
		return cnt == 0;
	}

	bool isFull() const {
		return cnt == Capacity - 1;
	}

	void resize(int newSize) {
		if (newSize < 0 || Capacity < newSize) {
			Exception("newSize out of range");
		}
		cnt = (SizeT)newSize;
	}

	void clear() {
		cnt = 0;
	}

	T& operator[](int index) {
		checkIndex(index);
		return buffer[index];
	}
	const T& operator[](int index) const {
		checkIndex(index);
		return buffer[index];
	}

	T& at(int index) {
		checkIndex(index);
		return buffer[index];
	}
	const T& at(int index) const {
		checkIndex(index);
		return buffer[index];
	}

	auto begin() {
		return buffer.begin();
	}
	auto begin() const {
		return buffer.begin();
	}
	auto end() {
		return buffer.begin() + cnt;
	}
	auto end() const {
		return buffer.begin() + cnt;
	}

	T& front() {
		checkEmpty();
		return buffer[0];
	}
	const T& front() const {
		checkEmpty();
		return buffer[0];
	}
	T& back() {
		checkEmpty();
		return buffer[cnt - 1];
	}
	const T& back() const {
		checkEmpty();
		return buffer[cnt - 1];
	}

	void remove(int index) {
		checkIndex(index);
		cnt--;
		for (int i = index; i < cnt; i++) {
			buffer[i] = buffer[i + 1];
		}
	}

	T take(int index) {
		T element = buffer[index];
		remove(index);
		return element;
	}

	static constexpr int capacity() {
		return Capacity;
	}

	T* data() {
		return &buffer[0];
	}
	const T* data() const {
		return &buffer[0];
	}

	void allClear() {
		buffer = {};
	}

	void outClear() {
		for (int i = cnt; i < std::ssize(buffer); i++) {
			buffer[i] = {};
		}
	}

protected:
	SizeT cnt;
	std::array<T, Capacity> buffer;

	void checkEmpty() const {
		if (cnt <= 0) {
			Exception("buffer empty");
		}
	}

	void checkIndex(int index) const {
		if (index < 0) {
			Exception("negative index " + std::to_string(index));
		}
		if (index >= cnt) {
			Exception("invalid index " + std::to_string(index) + " >= " + std::to_string(cnt));
		}
	}

	void checkFull() const {
		if (cnt >= Capacity) {
			Exception("buffer full. capacity:" + std::to_string(Capacity));
		}
	}
};

template<typename T, int Capacity>
using FixedList = FixedListBase<int, T, Capacity>;

template<typename T, int Capacity>
using ShortFixedList = FixedListBase<short, T, Capacity>;

template<typename T, int Capacity>
using ByteFixedList = FixedListBase<signed char, T, Capacity>;


constexpr std::ranges::iota_view<int, int> range(int count) {
	return std::ranges::iota_view<int, int>{ 0, count };
}

constexpr std::ranges::iota_view<int, int> range(size_t count) {
	return std::ranges::iota_view<int, int>{ 0, (int)count };
}

template<typename T>
constexpr std::ranges::iota_view<int, int> range(const std::vector<T>& list) {
	return std::ranges::iota_view<int, int>{ 0, (int)list.size() };
}

template<typename T, size_t Capacity>
constexpr std::ranges::iota_view<int, int> range(const std::array<T, Capacity>& ar) {
	return std::ranges::iota_view<int, int>{ 0, (int)ar.size() };
}

template<typename SizeT, typename T, SizeT Capacity>
constexpr std::ranges::iota_view<int, int> range(const FixedListBase<SizeT, T, Capacity>& list) {
	return std::ranges::iota_view<int, int>{ 0, list.size() };
}

template<typename TList, typename TElement>
inline bool Contains(const TList& list, const TElement& element) {
	for (const TElement& e : list) {
		if (e == element) {
			return true;
		}
	}
	return false;
}
