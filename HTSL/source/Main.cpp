#include <scene/Scene.h>

#include <ltbl/lighting/LightSystem.h>

#include <scene/SceneObjectMap.h>
#include <scene/SceneObjectGA.h>
#include <scene/SceneObjectFriendly.h>
#include <scene/SceneObjectLighting.h>
#include <scene/Overlay.h>

#include <random>

int main() {
	std::mt19937 generator(time(nullptr));

	sf::RenderWindow window;

	sf::ContextSettings contextSettings;

	window.create(sf::VideoMode(1280, 960), "LD32", sf::Style::Default, contextSettings);

	bool quit = false;

	sf::Clock clock;

	float dt = 0.017f;

	window.setVerticalSyncEnabled(true);

	window.setMouseCursorVisible(false);

	std::unique_ptr<Scene> scene = std::make_unique<Scene>();

	scene->create(window);

	scene->_generator.seed(time(nullptr));

	Ptr<SceneObjectLighting> lighting = make<SceneObjectLighting>();

	scene->add(lighting, "lighting");

	Ptr<SceneObjectMap> map = make<SceneObjectMap>();

	scene->add(map, "map");
	
	Ptr<SceneObjectGA> ga = make<SceneObjectGA>();

	scene->add(ga, "ga");

	Ptr<SceneObjectFriendly> friendly = make<SceneObjectFriendly>();

	friendly->_position = sf::Vector2f(200.0f, 200.0f);

	scene->add(friendly, "friendly");

	Ptr<Overlay> overlay = make<Overlay>();

	scene->add(overlay);

	do {
		clock.restart();

		// ----------------------------- Input -----------------------------

		sf::Event windowEvent;

		while (window.pollEvent(windowEvent)) {
			switch (windowEvent.type) {
			case sf::Event::Closed:
				quit = true;
				break;
			}
		}

		if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape))
			quit = true;

		window.clear();

		scene->update(dt);
		scene->render();

		// -------------------------------------------------------------------

		window.display();

		//dt = clock.getElapsedTime().asSeconds();
	} while (!quit);

	scene->destroyAll();

	scene->update(dt);

	return 0;
}