// This is the main DLL file.

#include "stdafx.h"

#include "ElektraDLL.h"
#include "Recognition.cpp"
#include <string>

namespace ElektraDLL {
	double Rodmenys::getRodmenis() {
		ContourWithData cwd = ContourWithData();
		return std::stod(cwd.init());
	}
}

