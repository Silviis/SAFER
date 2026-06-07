// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <mutex>
#include <optional>
#include <string>

#include <gz/common/Console.hh>
#include <gz/math/Vector3.hh>
#include <gz/msgs/twist.pb.h>
#include <gz/plugin/Register.hh>
#include <gz/sim/Link.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/System.hh>
#include <gz/transport/Node.hh>

namespace gz::sim::systems
{
class CmdVelFlightPlugin
		: public System,
			public ISystemConfigure,
			public ISystemPreUpdate
{
	public: CmdVelFlightPlugin() = default;

	public: void Configure(const Entity &_entity,
			const std::shared_ptr<const sdf::Element> &_sdf,
			EntityComponentManager &_ecm,
			EventManager &_eventMgr) override
	{
		static_cast<void>(_eventMgr);

		this->model = Model(_entity);

		if (_sdf && _sdf->HasElement("topic"))
		{
			this->topic = _sdf->Get<std::string>("topic");
		}

		if (_sdf && _sdf->HasElement("link_name"))
		{
			this->linkName = _sdf->Get<std::string>("link_name");
		}

		this->linkEntity = this->ResolveLinkEntity(_ecm);
		if (this->linkEntity == kNullEntity)
		{
			gzerr << "CmdVelFlightPlugin could not find a valid link for model '"
						<< this->model.Entity() << "'." << std::endl;
			return;
		}

		if (!this->node.Subscribe<CmdVelFlightPlugin, gz::msgs::Twist>(
			this->topic,
				&CmdVelFlightPlugin::OnCmdVel, this))
		{
			gzerr << "CmdVelFlightPlugin failed to subscribe to topic '"
						<< this->topic << "'." << std::endl;
			return;
		}

		this->configured = true;
		gzmsg << "CmdVelFlightPlugin listening on '" << this->topic
					<< "' for model '" << this->model.Entity() << "'." << std::endl;
	}

	public: void PreUpdate(const UpdateInfo &_info,
			EntityComponentManager &_ecm) override
	{
		if (!this->configured)
		{
			return;
		}

		const auto command = this->CurrentCommand();
		const auto link = gz::sim::Link(this->linkEntity);
		link.SetLinearVelocity(_ecm, command.linear);
		link.SetAngularVelocity(_ecm, command.angular);
	}

	private: struct VelocityCommand
	{
		gz::math::Vector3d linear{0.0, 0.0, 0.0};
		gz::math::Vector3d angular{0.0, 0.0, 0.0};
	};

	private: void OnCmdVel(const gz::msgs::Twist &_msg)
	{
		std::lock_guard<std::mutex> lock(this->mutex);
		this->command.linear.Set(
				_msg.linear().x(), _msg.linear().y(), _msg.linear().z());
		this->command.angular.Set(
				_msg.angular().x(), _msg.angular().y(), _msg.angular().z());
	}

	private: VelocityCommand CurrentCommand() const
	{
		std::lock_guard<std::mutex> lock(this->mutex);
		return this->command;
	}

	private: Entity ResolveLinkEntity(EntityComponentManager &_ecm) const
	{
		if (!this->linkName.empty())
		{
			const auto namedLink = this->model.LinkByName(_ecm, this->linkName);
			if (namedLink != kNullEntity)
			{
				return namedLink;
			}
		}

		const auto canonicalLink = this->model.CanonicalLink(_ecm);
		if (canonicalLink != kNullEntity)
		{
			return canonicalLink;
		}

		return this->model.LinkByName(_ecm, "base_link");
	}

	private: gz::transport::Node node;
	private: gz::transport::Node::Subscriber subscriber;
	private: gz::sim::Model model;
	private: gz::sim::Entity linkEntity{kNullEntity};
	private: std::string topic{"/world/surveillance_building/model/anafi4k/cmd_vel"};
	private: std::string linkName{"base_link"};
	private: bool configured{false};
	private: mutable std::mutex mutex;
	private: VelocityCommand command;
};
}

GZ_ADD_PLUGIN(gz::sim::systems::CmdVelFlightPlugin,
							gz::sim::System,
							gz::sim::ISystemConfigure,
							gz::sim::ISystemPreUpdate)

GZ_ADD_PLUGIN_ALIAS(gz::sim::systems::CmdVelFlightPlugin,
										"CmdVelFlightPlugin")
