#pragma once

#include <unordered_map>
#include <memory>
#include <string>

template<class T> using Ptr = std::shared_ptr<T>;

template<class T>
std::shared_ptr<T> make() {
	return std::make_shared<T>();
}

template<class T>
std::shared_ptr<T> make(T init) {
	return std::make_shared<T>(init);
}

class Table {
private:
	std::unordered_map<std::string, Ptr<void>> _data;

public:
	template<class T>
	void insert(const std::string &key, const T &value) {
		_data[key] = make<T>(value);
	}

	template<class T>
	void insertPtr(const std::string &key, const Ptr<T> &value) {
		_data[key] = value;
	}

	bool exists(const std::string &key) const {
		return _data.find(key) != _data.end();
	}

	template<class T>
	T &get(const std::string &key) {
		return *static_cast<T*>(_data[key].get());
	}

	template<class T>
	T* getPtr(const std::string &key) {
		return static_cast<T*>(_data[key].get());
	}

	void clear() {
		_data.clear();
	}
};