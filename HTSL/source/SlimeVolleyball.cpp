#include <Settings.h>

#if SUBPROGRAM_EXECUTE == SLIME_VOLLEYBALL

#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include <sc/HTSLPVLV.h>
#include <sc/HTSLSARSA.h>
#include <sc/HTSLQ.h>

#include <deep/FERL.h>

#include <iostream>

struct PhyObj {
	sf::Vector2f _position;
	sf::Vector2f _velocity;

	PhyObj()
		: _position(0.0f, 0.0f), _velocity(0.0f, 0.0f)
	{}
};

int main() {
	std::mt19937 generator(time(nullptr));

	sf::RenderWindow renderWindow;

	renderWindow.create(sf::VideoMode(1280, 720), "Reinforcement Learning", sf::Style::Default);

	renderWindow.setVerticalSyncEnabled(true);
	renderWindow.setFramerateLimit(60);

	// ---------------------------------- RL Init ------------------------------------

	/*sc::HTSLPVLV agentBlue;
	sc::HTSLPVLV agentRed;

	std::vector<sc::HTSLPVLV::InputType> inputTypes(20);

	for (int i = 0; i < 12; i++)
		inputTypes[i] = sc::HTSLPVLV::_state;
	
	for (int i = 12; i < 14; i++)
		inputTypes[i] = sc::HTSLPVLV::_action;

	for (int i = 14; i < 16; i++)
		inputTypes[i] = sc::HTSLPVLV::_pv;

	for (int i = 16; i < 18; i++)
		inputTypes[i] = sc::HTSLPVLV::_lve;

	for (int i = 18; i < 20; i++)
		inputTypes[i] = sc::HTSLPVLV::_lvi;

	std::vector<sc::HTSL::LayerDesc> layerDescs(2);

	layerDescs[0]._width = 20;
	layerDescs[0]._height = 20;

	layerDescs[1]._width = 12;
	layerDescs[1]._height = 12;

	//layerDescs[2]._width = 8;
	//layerDescs[2]._height = 8;

	agentBlue.createRandom(5, 4, 8, inputTypes, layerDescs, generator);
	agentRed.createRandom(5, 4, 8, inputTypes, layerDescs, generator);*/

	float rewardTimer = 0.0f;
	float rewardTime = 0.25f;
	float lastReward = 0.5f;

	//sc::HTSLSARSA agentBlue;
	sc::HTSLSARSA agentRed;

	std::vector<sc::HTSLSARSA::InputType> inputTypes(16);

	for (int i = 0; i < 12; i++)
		inputTypes[i] = sc::HTSLSARSA::_state;

	for (int i = 12; i < 14; i++)
		inputTypes[i] = sc::HTSLSARSA::_action;

	for (int i = 14; i < 16; i++)
		inputTypes[i] = sc::HTSLSARSA::_q;

	std::vector<sc::HTSL::LayerDesc> layerDescs(3);

	layerDescs[0]._width = 28;
	layerDescs[0]._height = 28;

	layerDescs[1]._width = 18;
	layerDescs[1]._height = 18;

	layerDescs[2]._width = 12;
	layerDescs[2]._height = 12;

	//agentBlue.createRandom(4, 4, 8, inputTypes, layerDescs, generator);
	agentRed.createRandom(4, 4, 8, inputTypes, layerDescs, generator);

	//deep::FERL agentBlue;
	//agentBlue.createRandom(12, 2, 32, 0.01f, generator);

	//deep::FERL agentRed;
	//agentRed.createRandom(12, 2, 32, 0.01f, generator);

	// --------------------------------- Game Init -----------------------------------

	const float slimeRadius = 94.5f;
	const float ballRadius = 23.5f;
	const float wallRadius = 22.5f;
	const float fieldRadius = 640.0f;

	const float gravity = 900.0f;
	const float slimeBounce = 100.0f;
	const float wallBounceDecay = 0.8f;
	const float slimeJump = 500.0f;
	const float maxSlimeSpeed = 1000.0f;
	const float slimeMoveAccel = 5000.0f;
	const float slimeMoveDeccel = 8.0f;

	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

	sf::Vector2f fieldCenter = sf::Vector2f(renderWindow.getSize().x * 0.5f, renderWindow.getSize().y * 0.5f + 254.0f);
	sf::Vector2f wallCenter = fieldCenter + sf::Vector2f(0.0f, -182.0f);

	PhyObj blue;
	PhyObj red;
	PhyObj ball;

	blue._position = fieldCenter + sf::Vector2f(-200.0f, 0.0f);
	blue._velocity = sf::Vector2f(0.0f, 0.0f);
	red._position = fieldCenter + sf::Vector2f(200.0f, 0.0f);
	red._velocity = sf::Vector2f(0.0f, 0.0f);
	ball._position = fieldCenter + sf::Vector2f(2.0f, -300.0f);
	ball._velocity = sf::Vector2f((dist01(generator)) * 600.0f, -(dist01(generator)) * 500.0f);

	sf::Texture backgroundTexture;
	backgroundTexture.loadFromFile("resources/slimevolleyball/background.png");

	sf::Texture blueSlimeTexture;
	blueSlimeTexture.loadFromFile("resources/slimevolleyball/slimeBodyBlue.png");

	sf::Texture redSlimeTexture;
	redSlimeTexture.loadFromFile("resources/slimevolleyball/slimeBodyRed.png");

	sf::Texture ballTexture;
	ballTexture.loadFromFile("resources/slimevolleyball/ball.png");

	sf::Texture eyeTexture;
	eyeTexture.loadFromFile("resources/slimevolleyball/slimeEye.png");
	eyeTexture.setSmooth(true);

	sf::Texture arrowTexture;
	arrowTexture.loadFromFile("resources/slimevolleyball/arrow.png");

	sf::Font scoreFont;
	scoreFont.loadFromFile("resources/slimevolleyball/scoreFont.ttf");

	int scoreRed = 0;
	int scoreBlue = 0;

	int prevScoreRed = 0;
	int prevScoreBlue = 0;

	float prevBallX = fieldCenter.x;

	// ------------------------------- Simulation Loop -------------------------------

	bool noRender = false;
	bool prevPressK = false;

	bool quit = false;

	float dt = 0.017f;

	do {
		sf::Event event;

		while (renderWindow.pollEvent(event)) {
			switch (event.type) {
			case sf::Event::Closed:
				quit = true;
				break;
			}
		}

		if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape))
			quit = true;

		// ---------------------------------- Physics ----------------------------------

		bool blueBounced = false;
		bool redBounced = false;

		// Ball
		{
			ball._velocity.y += gravity * dt;
			ball._position += ball._velocity * dt;

			// To floor (game restart)
			if (ball._position.y + ballRadius > fieldCenter.y) {
				if (ball._position.x < fieldCenter.x)
					scoreRed++;
				else
					scoreBlue++;

				blue._position = fieldCenter + sf::Vector2f(-200.0f, 0.0f);
				blue._velocity = sf::Vector2f(0.0f, 0.0f);
				red._position = fieldCenter + sf::Vector2f(200.0f, 0.0f);
				red._velocity = sf::Vector2f(0.0f, 0.0f);
				ball._position = fieldCenter + sf::Vector2f(2.0f, -300.0f);
				ball._velocity = sf::Vector2f((dist01(generator)) * 600.0f, -(dist01(generator)) * 500.0f);
			}

			// To wall
			if (((ball._position.x + ballRadius) > (wallCenter.x - wallRadius) && ball._position.x < wallCenter.x) || ((ball._position.x - ballRadius) < (wallCenter.x + wallRadius) && ball._position.x > wallCenter.x)) {
				// If above rounded part
				if (ball._position.y < wallCenter.y) {
					sf::Vector2f delta = ball._position - wallCenter;

					float dist = std::sqrt(delta.x * delta.x + delta.y * delta.y);

					if (dist < wallRadius + ballRadius) {
						sf::Vector2f normal = delta / dist;

						// Reflect velocity
						sf::Vector2f reflectedVelocity = ball._velocity - 2.0f * (ball._velocity.x * normal.x + ball._velocity.y * normal.y) * normal;

						ball._velocity = reflectedVelocity * wallBounceDecay;

						ball._position = wallCenter + normal * (wallRadius + ballRadius);
					}
				}
				else {
					// If on left side
					if (ball._position.x < wallCenter.x) {
						ball._velocity.x = wallBounceDecay * -ball._velocity.x;
						ball._position.x = wallCenter.x - wallRadius - ballRadius;
					}
					else {
						ball._velocity.x = wallBounceDecay * -ball._velocity.x;
						ball._position.x = wallCenter.x + wallRadius + ballRadius;
					}
				}
			}

			// To blue slime			
			{
				sf::Vector2f delta = ball._position - blue._position;

				float dist = std::sqrt(delta.x * delta.x + delta.y * delta.y);

				if (dist < slimeRadius + ballRadius) {
					sf::Vector2f normal = delta / dist;

					// Reflect velocity
					sf::Vector2f reflectedVelocity = ball._velocity - 2.0f * (ball._velocity.x * normal.x + ball._velocity.y * normal.y) * normal;

					float magnitude = std::sqrt(reflectedVelocity.x * reflectedVelocity.x + reflectedVelocity.y * reflectedVelocity.y);

					sf::Vector2f normalizedReflected = reflectedVelocity / magnitude;

					ball._velocity = blue._velocity + (magnitude > slimeBounce ? reflectedVelocity : normalizedReflected * slimeBounce);

					ball._position = blue._position + normal * (wallRadius + slimeRadius);

					blueBounced = true;
				}
			}

			// To red slime			
			{
				sf::Vector2f delta = ball._position - red._position;

				float dist = std::sqrt(delta.x * delta.x + delta.y * delta.y);

				if (dist < slimeRadius + ballRadius) {
					sf::Vector2f normal = delta / dist;

					// Reflect velocity
					sf::Vector2f reflectedVelocity = ball._velocity - 2.0f * (ball._velocity.x * normal.x + ball._velocity.y * normal.y) * normal;

					float magnitude = std::sqrt(reflectedVelocity.x * reflectedVelocity.x + reflectedVelocity.y * reflectedVelocity.y);

					sf::Vector2f normalizedReflected = reflectedVelocity / magnitude;

					ball._velocity = red._velocity + (magnitude > slimeBounce ? reflectedVelocity : normalizedReflected * slimeBounce);

					ball._position = red._position + normal * (wallRadius + slimeRadius);

					redBounced = true;
				}
			}

			// Out of field, left and right
			{
				if (ball._position.x - ballRadius < fieldCenter.x - fieldRadius) {
					ball._velocity.x = wallBounceDecay * -ball._velocity.x;
					ball._position.x = fieldCenter.x - fieldRadius + ballRadius;
				}
				else if (ball._position.x + ballRadius > fieldCenter.x + fieldRadius) {
					ball._velocity.x = wallBounceDecay * -ball._velocity.x;
					ball._position.x = fieldCenter.x + fieldRadius - ballRadius;
				}
			}
		}

		// Blue slime
		{		
			blue._velocity.y += gravity * dt;
			blue._velocity.x += -slimeMoveDeccel * blue._velocity.x * dt;
			blue._position += blue._velocity * dt;

			if (sf::Keyboard::isKeyPressed(sf::Keyboard::A)) {
				blue._velocity.x += -slimeMoveAccel * dt;

				if (blue._velocity.x < -maxSlimeSpeed)
					blue._velocity.x = -maxSlimeSpeed;
			}
			else if (sf::Keyboard::isKeyPressed(sf::Keyboard::D)) {
				blue._velocity.x += slimeMoveAccel * dt;

				if (blue._velocity.x > maxSlimeSpeed)
					blue._velocity.x = maxSlimeSpeed;
			}

			if (blue._position.y > fieldCenter.y) {
				blue._velocity.y = 0.0f;
				blue._position.y = fieldCenter.y;
				
				if (sf::Keyboard::isKeyPressed(sf::Keyboard::W))
					blue._velocity.y -= slimeJump;
			}

			if (blue._position.x - slimeRadius < fieldCenter.x - fieldRadius) {
				blue._velocity.x = 0.0f;
				blue._position.x = fieldCenter.x - fieldRadius + slimeRadius;
			}

			if (blue._position.x + slimeRadius > wallCenter.x - wallRadius) {
				blue._velocity.x = 0.0f;
				blue._position.x = wallCenter.x - wallRadius - slimeRadius;
			}
		}

		// Blue slime
		/*{
			const float scalar = 0.001f;
			// Percepts
			std::vector<float> inputs(12);

			int index = 0;

			inputs[index++] = (ball._position.x - blue._position.x) * scalar;
			inputs[index++] = (ball._position.y - blue._position.y) * scalar;
			inputs[index++] = ball._velocity.x * scalar;
			inputs[index++] = ball._velocity.y * scalar;
			inputs[index++] = (red._position.x - blue._position.x) * scalar;
			inputs[index++] = (red._position.y - blue._position.y) * scalar;
			inputs[index++] = red._velocity.x * scalar;
			inputs[index++] = red._velocity.y * scalar;
			inputs[index++] = blue._position.x * scalar;
			inputs[index++] = blue._position.y * scalar;
			inputs[index++] = blue._velocity.x * scalar;
			inputs[index++] = blue._velocity.y * scalar;

			std::vector<float> outputs(2);

			// Actions
			for (int i = 0; i < 12; i++)
				agentBlue.setState(i, inputs[i]);

			float reward = (ball._position.x > fieldCenter.x && prevBallX < fieldCenter.x ? 1.0f : 0.5f) * 0.5f + (scoreRed > prevScoreRed ? 0.0f : 0.5f) * 0.5f;

			if (reward != 0.5f) {
				lastReward = reward = reward > 0.5f ? 1.0f : 0.0f;
				rewardTimer = 0.0f;
			}
			else if (rewardTimer < rewardTime) {
				reward = lastReward;

				rewardTimer += dt;
			}

			if (blueBounced)
				reward = std::max(reward, 0.55f);

			reward = (reward * 2.0f - 1.0f) * 0.01f - std::abs(ball._position.x - blue._position.x) * 0.001f;

			//reward = (scoreBlue - prevScoreBlue) - (scoreRed - prevScoreRed) - std::abs(ball._position.x - blue._position.x) * 0.001f;

			agentBlue.update(reward, generator);

			float move = agentBlue.getActionFromNodeIndex(0) * 2.0f - 1.0f;
			bool jump = agentBlue.getActionFromNodeIndex(1) > 0.5f;

			//std::vector<float> action(2);
			//agentBlue.step(inputs, action, reward, 0.01f, 0.995f, 0.99f, 10.0f, 64, 3, 0.02f, 0.04f, 0.04f, 800, 200, 0.003f, 0.0f, generator);

			//float move = action[0];
			//bool jump = action[1] > 0.0f;

			blue._velocity.y += gravity * dt;
			blue._velocity.x += -slimeMoveDeccel * blue._velocity.x * dt;
			blue._position += blue._velocity * dt;

			{
				blue._velocity.x += move * slimeMoveAccel * dt;

				if (blue._velocity.x < -maxSlimeSpeed)
					blue._velocity.x = -maxSlimeSpeed;
				else if (blue._velocity.x > maxSlimeSpeed)
					blue._velocity.x = maxSlimeSpeed;
			}

			if (blue._position.y > fieldCenter.y) {
				blue._velocity.y = 0.0f;
				blue._position.y = fieldCenter.y;

				if (jump)
					blue._velocity.y -= slimeJump;
			}

			if (blue._position.x - slimeRadius < fieldCenter.x - fieldRadius) {
				blue._velocity.x = 0.0f;
				blue._position.x = fieldCenter.x - fieldRadius + slimeRadius;
			}

			if (blue._position.x + slimeRadius > wallCenter.x - wallRadius) {
				blue._velocity.x = 0.0f;
				blue._position.x = wallCenter.x - wallRadius - slimeRadius;
			}
		}*/

		// Red slime
		{
			const float scalar = 0.001f;
			// Percepts
			std::vector<float> inputs(12);

			int index = 0;

			inputs[index++] = (ball._position.x - red._position.x) * scalar;
			inputs[index++] = (ball._position.y - red._position.y) * scalar;
			inputs[index++] = ball._velocity.x * scalar;
			inputs[index++] = ball._velocity.y * scalar;
			inputs[index++] = (blue._position.x - red._position.x) * scalar;
			inputs[index++] = (blue._position.y - red._position.y) * scalar;
			inputs[index++] = red._velocity.x * scalar;
			inputs[index++] = red._velocity.y * scalar;
			inputs[index++] = blue._position.x * scalar;
			inputs[index++] = blue._position.y * scalar;
			inputs[index++] = blue._velocity.x * scalar;
			inputs[index++] = blue._velocity.y * scalar;

			//std::vector<float> outputs(2);

			// Actions
			for (int i = 0; i < 12; i++)
				agentRed.setState(i, inputs[i]);

			float reward = (ball._position.x < fieldCenter.x && prevBallX >= fieldCenter.x ? 1.0f : 0.5f);
			
			if (reward != 0.5f) {
				lastReward = reward = reward > 0.5f ? 1.0f : 0.0f;
				rewardTimer = 0.0f;
			}
			else if (rewardTimer < rewardTime) {
				reward = lastReward;

				rewardTimer += dt;
			}

			if (redBounced)
				reward = std::max(reward, 0.55f);

			reward = (reward * 2.0f - 1.0f) * 0.1f - std::abs(ball._position.x - red._position.x) * 0.008f;

			//reward *= 0.05f;

			//reward = (scoreRed - prevScoreRed) - (scoreBlue - prevScoreBlue) - std::abs(ball._position.x - red._position.x) * 0.001f;

			//std::cout << "Reward: " << reward << std::endl;

			//agentRed.update(reward, generator);
			agentRed.update(reward, generator);
			//std::vector<float> action(2);
			//agentRed.step(inputs, action, reward, 0.01f, 0.995f, 0.99f, 10.0f, 32, 6, 0.05f, 0.04f, 0.04f, 800, 200, 0.003f, 0.0f, generator);

			//float move = action[0];
			//bool jump = action[1] > 0.0f;

			float move = agentRed.getActionFromNodeIndex(0) * 2.0f - 1.0f;
			bool jump = agentRed.getActionFromNodeIndex(1) > 0.5f;

			red._velocity.y += gravity * dt;
			red._velocity.x += -slimeMoveDeccel * red._velocity.x * dt;
			red._position += red._velocity * dt;

			{
				red._velocity.x += move * slimeMoveAccel * dt;

				if (red._velocity.x < -maxSlimeSpeed)
					red._velocity.x = -maxSlimeSpeed;
				else if (red._velocity.x > maxSlimeSpeed)
					red._velocity.x = maxSlimeSpeed;
			}
			
			if (red._position.y > fieldCenter.y) {
				red._velocity.y = 0.0f;
				red._position.y = fieldCenter.y;

				if (jump)
					red._velocity.y -= slimeJump;
			}

			if (red._position.x + slimeRadius > fieldCenter.x + fieldRadius) {
				red._velocity.x = 0.0f;
				red._position.x = fieldCenter.x + fieldRadius - slimeRadius;
			}

			if (red._position.x - slimeRadius < wallCenter.x + wallRadius) {
				red._velocity.x = 0.0f;
				red._position.x = wallCenter.x + wallRadius + slimeRadius;
			}
		}

		prevScoreRed = scoreRed;
		prevScoreBlue = scoreBlue;
		prevBallX = ball._position.x;

		if (sf::Keyboard::isKeyPressed(sf::Keyboard::K) && !prevPressK) {
			noRender = !noRender;
		}

		prevPressK = sf::Keyboard::isKeyPressed(sf::Keyboard::K);

		if (noRender)
			continue;

		// --------------------------------- Rendering ---------------------------------

		renderWindow.clear();

		{
			sf::Sprite s;
			s.setTexture(backgroundTexture);
			s.setOrigin(backgroundTexture.getSize().x * 0.5f, backgroundTexture.getSize().y * 0.5f);
			s.setPosition(renderWindow.getSize().x * 0.5f, renderWindow.getSize().y * 0.5f);

			renderWindow.draw(s);
		}

		{
			sf::Sprite s;
			s.setTexture(blueSlimeTexture);
			s.setOrigin(blueSlimeTexture.getSize().x * 0.5f, blueSlimeTexture.getSize().y);
			s.setPosition(blue._position);

			renderWindow.draw(s);
		}

		{
			sf::Sprite s;
			s.setTexture(eyeTexture);
			s.setOrigin(eyeTexture.getSize().x * 0.5f, eyeTexture.getSize().y * 0.5f);
			s.setPosition(blue._position + sf::Vector2f(50.0f, -28.0f));

			sf::Vector2f delta = ball._position - s.getPosition();

			float angle = std::atan2(delta.y, delta.x);

			s.setRotation(angle * 180.0f / 3.141596f);

			renderWindow.draw(s);
		}

		{
			sf::Sprite s;
			s.setTexture(redSlimeTexture);
			s.setOrigin(redSlimeTexture.getSize().x * 0.5f, redSlimeTexture.getSize().y);
			s.setPosition(red._position);

			renderWindow.draw(s);
		}

		{
			sf::Sprite s;
			s.setTexture(eyeTexture);
			s.setOrigin(eyeTexture.getSize().x * 0.5f, eyeTexture.getSize().y * 0.5f);
			s.setPosition(red._position + sf::Vector2f(-50.0f, -28.0f));

			sf::Vector2f delta = ball._position - s.getPosition();

			float angle = std::atan2(delta.y, delta.x);

			s.setRotation(angle * 180.0f / 3.141596f);

			renderWindow.draw(s);
		}

		{
			sf::Sprite s;
			s.setTexture(ballTexture);
			s.setOrigin(ballTexture.getSize().x * 0.5f, ballTexture.getSize().y * 0.5f);
			s.setPosition(ball._position);

			renderWindow.draw(s);
		}

		if (ball._position.y + ballRadius < 0.0f) {
			sf::Sprite s;
			s.setTexture(arrowTexture);
			s.setOrigin(arrowTexture.getSize().x * 0.5f, 0.0f);
			s.setPosition(ball._position.x, 0.0f);

			renderWindow.draw(s);
		}

		{
			sf::Text scoreText;
			scoreText.setFont(scoreFont);
			scoreText.setString(std::to_string(scoreBlue));
			scoreText.setCharacterSize(100);

			float width = scoreText.getLocalBounds().width;

			scoreText.setPosition(fieldCenter.x - width * 0.5f - 100.0f, 10.0f);
			
			scoreText.setColor(sf::Color(100, 133, 255));

			renderWindow.draw(scoreText);
		}

		{
			sf::Text scoreText;
			scoreText.setFont(scoreFont);
			scoreText.setString(std::to_string(scoreRed));
			scoreText.setCharacterSize(100);

			float width = scoreText.getLocalBounds().width;

			scoreText.setPosition(fieldCenter.x - width * 0.5f + 100.0f, 10.0f);

			scoreText.setColor(sf::Color(255, 100, 100));
			
			renderWindow.draw(scoreText);
		}

		if (sf::Keyboard::isKeyPressed(sf::Keyboard::P)) {
			float scale = 4.0f;

			float alignment = 0.0f;

			/*for (int l = 0; l < agentBlue.getHTSL().getLayers().size(); l++) {
				sf::Image sdr;
				sdr.create(agentBlue.getHTSL().getLayerDescs()[l]._width, agentBlue.getHTSL().getLayerDescs()[l]._height);

				for (int x = 0; x < agentBlue.getHTSL().getLayerDescs()[l]._width; x++)
					for (int y = 0; y < agentBlue.getHTSL().getLayerDescs()[l]._height; y++) {
						sf::Color c;
						c.r = c.g = c.b = agentBlue.getHTSL().getLayers()[l]._rsc.getHiddenState(x, y) * 255.0f;

						sdr.setPixel(x, y, c);
					}

				sf::Texture sdrt;
				sdrt.loadFromImage(sdr);

				sf::Sprite sdrs;
				sdrs.setTexture(sdrt);
				sdrs.setPosition(alignment, renderWindow.getSize().y - sdr.getSize().y * scale);
				sdrs.setScale(scale, scale);

				renderWindow.draw(sdrs);

				alignment += scale * sdr.getSize().x;
			}*/

			alignment = 0.0f;

			for (int l = 0; l < agentRed.getHTSL().getLayers().size(); l++) {
				sf::Image sdr;
				sdr.create(agentRed.getHTSL().getLayerDescs()[l]._width, agentRed.getHTSL().getLayerDescs()[l]._height);

				for (int x = 0; x < agentRed.getHTSL().getLayerDescs()[l]._width; x++)
					for (int y = 0; y < agentRed.getHTSL().getLayerDescs()[l]._height; y++) {
						sf::Color c;
						c.r = c.g = c.b = agentRed.getHTSL().getLayers()[l]._rsc.getHiddenState(x, y) * 255.0f;

						sdr.setPixel(x, y, c);
					}

				sf::Texture sdrt;
				sdrt.loadFromImage(sdr);

				alignment += scale * sdr.getSize().x;

				sf::Sprite sdrs;
				sdrs.setTexture(sdrt);
				sdrs.setPosition(renderWindow.getSize().x - alignment, renderWindow.getSize().y - sdr.getSize().y * scale);
				sdrs.setScale(scale, scale);

				renderWindow.draw(sdrs);	
			}
		}

		if (sf::Keyboard::isKeyPressed(sf::Keyboard::T)) {
			sf::Vector2f position;

			const float scalar = 1.0f / 0.001f;

			position.x = red._position.x + scalar * agentRed.getHTSL().getPrediction(0);
			position.y = red._position.y + scalar * agentRed.getHTSL().getPrediction(1);

			sf::Sprite s;
			s.setTexture(ballTexture);
			s.setOrigin(ballTexture.getSize().x * 0.5f, ballTexture.getSize().y * 0.5f);
			s.setPosition(position);

			renderWindow.draw(s);
		}

		if (sf::Keyboard::isKeyPressed(sf::Keyboard::F)) {
			for (int i = 0; i < 16; i++)
				std::cout << agentRed.getHTSL().getPrediction(i) << std::endl;

			std::cout << std::endl;
		}
		
		renderWindow.display();
	} while (!quit);

	return 0;
}

#endif