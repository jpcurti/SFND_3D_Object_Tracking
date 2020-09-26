from conans import ConanFile, CMake

class ObjectTrackingConan(ConanFile):
	settings = "os", "compiler", "build_type", "arch"
	requires = "opencv/4.1.1@conan/stable"
	generators = "cmake", "cmake_find_package"
	default_options = {"opencv:nonfree" : True}
	def imports(self):
		self.copy("*.dll", dst="bin", src="bin")
		self.copy("*.dylib*", dst="bin", src="lib")
	def build(self):
		cmake = CMake(self)
		cmake.configure()
		cmake.build()