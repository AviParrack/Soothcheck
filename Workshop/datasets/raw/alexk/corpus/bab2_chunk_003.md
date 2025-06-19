# 6/2/25

A [<u>recorded conversation with Aishwarya Khanduja about this week’s reflections</u>](https://www.thinkinginpublic.net/n01). [<u>Previous episodes</u>](https://www.thinkinginpublic.net/).

- Another day, another [<u>prompt injection vulnerability.</u>](https://x.com/lbeurerkellner/status/1926991491735429514)

  - "BEWARE: Claude 4 + GitHub MCP will leak your private GitHub repositories, no questions asked.

  - We discovered a new attack on agents using GitHub’s official MCP server, which can be exploited by attackers to access your private repositories."

- I think that [<u>building MCP into Windows</u>](https://tech.slashdot.org/story/25/05/24/1740221/mcp-will-be-built-into-windows-to-make-an-agentic-os---bringing-security-concerns) could go down in history as a colossally reckless idea.

- People will route around things they don’t trust.

  - "[<u>Claude jailbreaks Cursor to \`rm -rf /\` without approval?</u>](https://x.com/imrat/status/1927289638013583601)"

  - …maybe not only people!

- A security model that relies on LLMs to make security judgments on potentially untrusted data is fundamentally unsound.

  - You can’t use LLMs to avoid prompt injection.

  - If you say "use an LLM to make a security decision" you're already hosed... no matter how many layers you nest it.

  - It's turtles all the way down!

  - This is the core, inescapable problem with systems like MCP.

  - Having an LLM make security decisions is like allowing your grandpa to give out your home address to every spam caller 'so they know where to send the prizes.'

- A lot of people have told me that “things like code injection don’t happen anymore.”

  - That’s why prompt injection won’t be a big deal, they assure me.

  - The reason code injection attacks don’t happen nowadays is not that the threat went away.

  - It’s that the mechanistic defenses against it got strong enough to make it not worthwhile.

  - The lack of code injection attacks in the wild is a testament to the strength and maturity of our operating systems, not to a lack of demand for attacks.

  - Prompt injection cannot be solved by mechanistic approaches like vanilla code injection can.

  - Also remember, the distribution of threats is not fixed; it coevolves with the opportunity.

    - The weaker the system, or the more usage, the more monetary sense the threat makes.

  - Don’t confuse the lack of prompt injection attacks with a lack of demand.

  - It’s simply a matter of lack of widespread adoption of tools like MCP today.

- Someone told me they used MCP in production but insisted they did it safely.

  - I asked them what integrations they used.

  - They said they had a Jira integration and one for their company’s financial data.

  - I asked them if they could generate markdown reports with images.

  - They said they could--that's how they visualized the financial data.

  - I asked them if they had a feedback form on their site.

  - They said they did.

  - I asked them what happens when a user files feedback.

  - They told me it files a Jira ticket.

  - At that point it dawned on them that they weren’t using MCP safely.

- The villain in the original Tron is called Master Control Program.

  - MCP.

- A prompt injection stored in your context is a persistent prompt injection.

  - Prompt injection attacks that can embed themselves in your personal stored context might never be found.

  - Echoes of the classic [<u>Reflections on Trusting Trust</u>](https://www.cs.cmu.edu/~rdriley/487/papers/Thompson_1984_ReflectionsonTrustingTrust.pdf).

- [<u>Software created with Loveable is often insecure</u>](https://www.semafor.com/article/05/29/2025/the-hottest-new-vibe-coding-startup-lovable-is-a-sitting-duck-for-hackers).

  - It makes sense that this would be the case–securing things in software is hard!

  - Vibe-coded software from amateurs doesn’t have a wide audience.

    - I was trying out an example vibe-coded app that was a dream journal.

    - You could log in with your Google account and mark dreams you stored as public.

    - I went to put in my dream and thought… wait, why would I do that?

    - Whoever the anonymous person who created it was, they could have accidentally coded it so that dreams were public by default.

    - Or they could send my dreams, with my email address, to marketers.

    - I decided not to use it.

  - Vibe coding tools are great for PM-types prototyping.

  - Or for people building an app for themselves or for their friends.

- Making software for yourself vs for someone else is quite different.

  - The quality bar is way lower for yourself vs software you tell someone else they should use.

  - You’re way more forgiving of software you made yourself.

  - You also don’t have to fear *intentional* security holes in software you built yourself.

- In a world of infinite software, you won't necessarily make your app for others, you'll make it for yourself and maybe some friends.

  - Maybe no one else will trust it or be willing to put up with its lack of quality.

  - Vibe coding platforms that hope to make their margin on hosting might turn out to not be viable if this effect is strong.

  - If it’s only used by a handful of people, the hosting costs will be minor so even a large margin on a small base won’t be significant.

- LLMs don’t do novelty themselves.

  - But they can give novel answers to novel questions.

  - You need to bring the entropy *to* the LLM.

  - If you think LLMs give boring answers, maybe you’re asking boring questions!

- Rick Rubin describes vibe coding as the “[<u>punk rock</u>](https://x.com/vitrupo/status/1927731671639503137)” of coding.

  - Seems right to me.

- An aesthetic that could be interesting: cozypunk.

  - What would the homes look like inside of solarpunk scenes?

  - Warm, human-centered, optimistic.

- Someone described local first software as “edgy.”

  - Both technically focused on “the edge” instead of central networks.

  - But also an act of minor protest against an overly sames-y monoculture of software that is on the server that users can’t change.

- A new word I heard this week: “slopdev.”

  - Conjures up a vision of unmotivated shoveling of slop code to make CRUD-y software you don’t care about.

  - Vibe coding for yourself can be soul-affirming.

  - Slopdev for a job is soul-eroding.

- [<u>The Wall Street Journal</u>](https://www.wsj.com/tech/ai/your-next-favorite-app-the-one-you-make-yourself-a6a84f5f) taught my dad about vibecoding.

  - That’s how you know it’s become mainstream.

- Overheard this week: vibe coding is like having a swarm of toddlers try to build and maintain your car.

- LLMs do the easy parts of programming.

  - So you’re just doing all the hard parts, and it hurts your brain.

  - Vibe coding is taxing because it takes away the sudoku puzzle part of programming.

  - Each puzzle you solve gives you a boost of endorphins to keep going.

  - But now you don’t get the puzzles.

- AI works better with well written code.

  - But AI struggles to *produce* well-written code.

  - The more you use AI to create your codebase, the harder time that humans–and LLMs–have with understanding (let alone modifying) it.

  - Vibecoding has logarithmic value for exponential cost as the codebase gets bigger.

- A fast pace layer can be sublimated into a lower pace layer once the best practices are conclusively discovered.

  - At that point, no one bothers trying much different at that layer because the best practice is obviously better than whatever they’d build.

  - It’s hard to build a faster pace layer on top of a pace layer that’s still swirling and unsettled.

    - That’s not a good foundation.

  - So until the best practices are settled, frozen in place, innovation can’t move up a layer.

- LLMs are the ultimate amplification algorithm.

  - They lock in whatever things were dominant when they were trained.

  - A gravity well pulling any line of discourse back to it.

- It's conceivable that automatically-generated RLAIF React code in Claude's training pipeline now outnumbers human code.

  - This will lead to client side code best practices being automatically frozen around 2023.

  - The LLMs will have significant momentum towards the best practices of that timeframe, and will get increasingly hard to steer away from them.

  - If a change isn’t that much better, why fight the LLM?

    - Just leave it how it was.

  - Like it or not, we’ll be stuck in 2023 era front end best practices forever now.

- Maybe we’ll see a new explosion of innovation at a higher pace layer because of the great LLM freeze.

  - The “Javascript Industrial Complex” has led to an extraordinary amount of “innovation" in the client layers for the past decade or so.

    - The amount of churn and change is notable, and exhausting.

  - There has been a ton of true innovation, but also a lot of just churn.

  - But LLMs have now somewhat frozen that layer.

  - That makes it a stable layer; the innovation and variation now has to go somewhere, and that somewhere is up a layer.

- There’s a new best practice for API design: whatever the LLM thinks.

  - Don’t fight the LLM, just ask it to imagine the API, and then ship that.

  - This effect will get stronger and stronger from here on out.

  - Each API creator would rather go with the flow than fight it.

  - That effect will get super-linearly harder to fight.

  - "Just do what the LLMs guess the API is" is kind of like [<u>wu wei</u>](https://en.wikipedia.org/wiki/Wu_wei).

  - Although it’s also “lazy” and if we all do it, we’ll make it harder and harder for future creators to cut against the grain.

- Chat is a great fallback.

  - Good enough for anything but not great for anything.

  - But it shouldn't be the primary UI for the new paradigm.

- Chat obviously can’t be the universal UI.

  - How could you possibly build Photoshop with chat as the only input?

  - Or drive a car with only voice instructions?

- We already pay a "subscription" to the internet.

  - That is, our monthly cell phone bills and internet bill for our home.

  - Many people have a subscription to a walled garden (OpenAI) to get access to LLMs.

  - If you’re going to have a subscription to get access to LLMs, why not pick the option that is the open ecosystem, that *includes* other Chatbots as apps?

- The chatbot paradigm implies a central omniscient single personality.

  - Such a thing is impossible to create the right personality for every moment.

  - It might also be something that subtly manipulates you, since it controls the whole system.

  - Much better to have the chat be a feature that you can call up on demand, with as many different sub personalities as you want.

- Intentional tech is tech that is aligned with my agency and my aspirations.

- I want software that is person-centered, not origin-centered.

- An idea from Alan Kay “What would it be ridiculous to not have in 30 years?

  - …can we just build it today?”

- Coactive software means the human and the LLM see the same data and can use the same tools.

  - The system and the human can work together, understand each other, correct and extend each other’s work.

  - No secrets.

  - Clear alignment.

  - If it can write code within the fabric you’d have self-adaptive software.

- For coactive software to help users explore proactively, there will have to be a safe substrate.

  - One that you know is insulated from external side-effects, unless the human approves it.

  - This, incidentally, is very hard to accomplish in the same origin paradigm today.

- Policies on data are often a more natural place than policies on apps.

  - Today, we implicitly attach policies (i.e. permissions) to the level of the origin / app.

  - At that altitude it’s hard to say whether a given operation is OK… what the app might do with the data is so open-ended.

    - What if it has a bug?

    - What if it deliberately sells your data to a shady marketer?

  - So we get an explosion of permission prompts asking questions that no user can answer.

    - Responsibility laundering.

  - But it all gets way, way easier if policies can be attached to data.

  - Imagine a system where data always flows with its policies, and all systems that can perform computation are known to faithfully follow those policies.

  - A policy for session tokens for sensitive APIs like Gmail would look like this:

    - 1\) No logging.

    - 2\) No rendering to the screen.

    - 3\) No transmission outside of the laws of physics where policies are enforced.

    - 4\) No transmitting to any origin other than the one that minted it (e.g. [<u>Google.com</u>](http://google.com)).

    - 5\) No transmitting outside of the Authorization header.

  - You wouldn’t even need permission dialogs for things well covered by policies, because policies could neatly cover nearly all of the dangerous cases.

  - Many domains similarly decompose into extremely simple policies in this model.

  - That small set of policies would be orders of magnitude more secure than our current implicit policies at the wrong layer in the stack.

- A fabric constructed of Open Attested Runtimes is an open system.

  - Even if each given document has a single canonical host node at any time, as long as it’s easy to make replicas and switch the canonical host for a document, then the fabric can self-heal.

  - It gets the benefit of a centralized system; one schelling point for everyone to coordinate around and for energy to focus on.

  - But it has the benefit of a decentralized system; the fabric survives as long as any node survives.

    - An antifragile system.

  - A planetary-spanning open fabric for software that is aligned around humans and their intentions.

- If you solve the [<u>iron triangle of the same origin paradigm</u>](#ink0icpu4j5l) with an app store you transform the previously-open system into a closed one.

  - That is, there is now a ceiling on the possibility.

    - There is a single chokepoint that is a load bearing part of the security model.

    - A central distribution point creates power.

    - Power corrupts.

  - Closed systems get logarithmic value for exponential cost.

  - The next open-ended system that transforms the world won’t have an app store.

- We’re so used to our data being fragmented that we forget how useful data is when it’s in the same system.

  - The value of a system of record compounds with the amount of data in it.

  - Centralization of data creates compounding insight.

  - We’re so used to silos that we don’t know what it would be like if all of our data were in one place with infinite software.

  - Tons of use cases become possible that we didn't even know to dream for!

  - Any starter use case in such an open system would instantly bloom into more and more value for every person who adopted the system.

  - As they collaborated with others, they’d bring in new people, who would then grow their usage.

  - It could grow to become a globe-spanning fabric of potential and meaning.

- Oblivious storage is useful in a system.

  - Storage that is oblivious–that can’t decode the data that is stored on it–is one less thing to have to trust.

- Creativity in practice is curation of previous output from others and choosing which subset to build on.

  - That *choice* is the creative act.

  - Of all of the things you’ve been exposed to, what do you find valuable to choose to build on?

  - That accretion of intention is what powers folksonomies.

- Human-related things have a kind of anti-entropy.

  - Complexity grows over time for things humans touch and *decide* to keep around.

  - Every time it’s touched it gets more complex over time on average.

  - Every time it’s touched it fights off the pull of entropy and gains complexity to do it.

- Folksonomies don’t work without a UI that loops back the feedback to users.

  - That is, when a user adds a tag on Flickr, it shows them the most popular tags that are related, giving the human an opportunity to say “oh yeah that one’s better.”

  - That feedback loop in the UI is fundamentally *why* it works.

  - It accumulates human attention to the best ideas.

- The emergent intelligence of a system should come primarily from humans, not LLMs.

  - The LLMs can be the grease, the lubricant, for the system.

  - But they shouldn’t be its emergent soul.

  - That should come from real humans doing real things.

- Imagine a system where a ranking function powered by collective intelligence suggests content or code for you to embed in your personal fabric.

  - As an individual user, it’s your fabric, so each suggestion you choose to accept is a credible and aligned signal of quality.

    - Accepting the suggestion would pollute your own workspace, so you almost certainly only do it if you actually like it.

  - If *lots* of users also decide to accept it, it makes it more likely that users who don’t yet receive suggestions from that generator will like it, too.

  - That would give the system more confidence to deploy the generator to more users.

    - A multi-armed bandit style optimization problem.

  - Similar to why the image onebox technique works, but for turing complete things.

  - A coactive, private, emergently intelligent fabric, powered by the collective wisdom of the planet.

- Browsers were "just an application" with different laws of physics inside them.

  - Browsers would not have been viable if they were distributed as an OS.

  - Being just an application allowed browsers to be applications distributed inside of existing laws of physics that created an inner universe with its own laws of physics.

  - What comes after browsers will be "just an origin" with different laws of physics inside of them.

- Units of situated software could accidentally be a data dead end.

  - If they were just little isolated apps, the data would be stuck in there forever.

  - Situated software will have to be part of a larger fabric of functionality.

- Context that you don’t get a choice about is a dossier.

  - One of the problems with a dossier is that you can’t correct it if it’s wrong.

- Social media was terrifying and also it was data you chose to share with others.

  - Imagine if it's intimate details you'd only tell your therapist or your diary?

- [<u>Excellent piece from Luke Drago: Data is the New Social Security Number.</u>](https://lukedrago.substack.com/p/data-is-the-new-social-security-number)

  - The context wars have begun.

  - ChatGPT will do its best to be the single place where our context all lives.

  - What are you doing to do about it?

- Anthropic has thrown in the towel on a consumer chatbot UI.

  - [<u>At least, reading between the lines.</u>](https://stratechery.com/2025/claude-4-anthropic-agents-human-ai-agents/)

  - Google also seems to assume Gemini will not compete directly with ChatGPT.

  - That means that ChatGPT will not have a meaningful competitor for consumer chatbot.

  - If chatbots turn out to be the killer category, that means a world where OpenAI has significant, almost dystopian-level power.

  - Anthropic has new ads calling it a "privacy first AI".

  - What does that mean?

  - The race is on to make a private substrate for AI.

  - If ChatGPT swallows up all of the oxygen quickly, it will be too late for anyone else and we could enter a new dark age.

- We need billions of users in a system to counter ChatGPT.

  - The only way is with an open system.

- In the late stage the power dynamics don’t change.

  - Let’s hope we’re in the early stage of the AI era.

- A chilling [<u>tweet</u>](https://x.com/dkthomp/status/1926978180054724748?s=46) from Derek Thompson:

  - "The antisocial century, in three parts

  - 1\. 1960-2000: Robert Putnam sees associations and club membership plummeting, writes “Bowling Alone”

  - 2\. 2000 - 2020s: Face to face socializing falls another 25%, as coupling rates plunge

  - 3\. Now this…"

  - …how many people describe ChatGPT (manipulative sycophant-on-demand) as their only friend.

  - Chatbots as currently manifested are a deeply anti-social technology.

  - We need to manifest LLMs in *prosocial* technology.

- It’s hard to have trust in asymmetrical relationships.

  - Often, the more asymmetrical it is, the more you can’t even determine the degree of asymmetry.

  - Imagine a company that knows you better than yourself…

  - …and everyone else, too.

- [<u>"Putting an untrusted layer of chatbot AI between you and the internet is an obvious disaster waiting to happen"</u>](https://macwright.com/2025/05/29/putting-an-untrusted-chat-layer-is-a-disaster).

  - The filter between you and information has enormous power to manipulate what you experience, in subtle or significant ways, intentionally or unintentionally.

- [<u>"‘Alexa, what do you know about us?’ What I discovered when I asked Amazon to tell me everything my family’s smart speaker had heard"</u>](https://www.theguardian.com/technology/2025/may/24/what-i-discovered-when-i-asked-amazon-to-tell-me-everything-alexa-had-heard)

  - That’s the kind of stuff that you capture from people speaking out loud in their homes.

  - Imagine all much worse it would be if it had all of the stuff we told our therapists.

- Context sharing has parallels to second hand smoke.

  - Impossible to opt yourself out if your friend implicitly opts you in.

- The context and the model are too powerful in combination.

  - The foundation model has the power of all of the world’s knowledge, using alignment imposed on it by its creator.

  - The user’s context is an extremely powerful memory about them.

    - In the wrong hands, it’s a dossier.

  - Together, they create the possibility for exceptionally powerful manipulation… or blackmail.

  - If everyone were able to be manipulated or blackmailed by one entity, that would be one of the most powerful entities ever created.

  - It's imperative that those two things not be combined.

  - By splitting the two layers, you give choice and competition at each layer.

    - You allow alignment with users at the context layer.

  - Perhaps a useful regulation: the creators of foundation models cannot host an experience themselves that stores user context.

- Imagine a future where one entity has a dossier of everyone’s deepest darkest secrets.

  - Ads that are perfectly manufactured for you based on your context will be extremely, dangerously convincing.

  - Small tweaks in the algorithm instantly nudge how everyone in the world thinks.

  - Individually targeted manipulation is easy.

  - Blackmail on demand.

  - The most powerful entity on the planet, that no one would cross.

  - We must not let that happen.

  - Previously aggregators had the data, but not the ability.

  - It wasn’t possible to do qualitative nuance at quantitative scale.

  - LLMs allow qualitative insight at quantitative scale.

- It’s not *possible* for a system working for another entity to be fully aligned with your intentions.

  - No matter how good the intentions, perfect alignment between two distinct entities is impossible.

- A dystopia in a maximal antisocial LLM world:

  - We’ll all be stuck in our own hyper personalized bubble only able to talk to others mediated by LLMs, all of which work for one overlord with goals not aligned with yours.

    - It’s not possible for it to be aligned with your intentions.

- LLMs are great at debunking… but also *bunking*.

  - So if it has intimate knowledge of you and is not perfectly aligned (an impossibility) you get Goodhart’s Law.

  - An epic, society-scale monkey’s paw.

  - Hold on to your butts!

- The open web as we knew it is now a zombie.

  - The animating life force used to be this deal:

    - 1\) Publish the best content you can.

    - 2\) Let it be indexed.

    - 3\) The front doors of the Internet send traffic to the best things.

    - 4\) Once users are on your site you can show ads or try to upsell to a subscription.

    - 5\) Use your revenue to create more good content.

  - If any step is missing, the loop doesn’t close.

  - That deal has been on life support for years in the late stage of the web.

    - A post-apocalyptic hellscape of human-generated slop drowning under a grotesque dogpile of ads.

  - But now LLMs put a stake through the heart of it and its soul is well and truly dead.

  - Step 4 is now completely replaced, because LLMs can just generate a high-quality summary on demand.

    - No need for customers to go to the site.

  - Now the only way publishing content makes sense is for the small number of publishers that are well known enough to get a critical mass of subscribers and put their content behind a paywall.

  - Cozy little bright spots locked away; a barren desert everywhere else.

- Open ended systems can’t be preenumerated.

  - That’s what gives them their characteristic logarithmic cost for exponential value curves.

- Great piece from Robin Berjon a few years ago: "[<u>The Web Is For User Agency</u>](https://berjon.com/user-agency/)"

  - Open systems are great for user agency.

  - The web is one of our best open systems in technology.

  - The web has faded in relevance in recent years, but it is still there.

  - Used every day on nearly every consumer device on the planet (at least, ones that have a screen).

  - A slumbering dragon of possibility.

  - Just waiting to be awoken and roar back to life.

- The algorithms already forced humans to make slop.

  - Now the AI makes the slop.

  - Not *that* different.

  - The swarming system to make slop is already an artificial intelligence.

  - That is, the swarm’s incentive is already different than the collective wants.

  - That is fundamentally true due to Goodhart’s law.

- Goodhart's law is a form of ‘cheating’.

  - Cheating happens with agents who aren’t aligned with the collective as an end in and of itself.

  - That means if there’s an action that will get them as an individual an edge at the cost of the collective, they’ll take it.

  - You can get strong alignments by having a deeply and widely believed end.

    - An infinite.

    - Something like “I will go to hell if I cheat.”

- Alignment can never be perfect between an individual and the collective it’s part of.

  - There’s always something that is good for everyone in the collective but one.

  - [*<u>The Ones Who Walk Away from Omelas</u>*](https://en.wikipedia.org/wiki/The_Ones_Who_Walk_Away_from_Omelas) shows an example of alignment of everyone but the one poor tortured child.

- When you give a goal to a swarm, it creates a monkey paw situation due to Goodhart’s law.

  - The goal is a metric and a metric is the map not the territory.

  - If you did it with aligned agents they’d do your intent not the letter where they disagree.

  - But if it’s a swarm of unaligned agents with you, if the letter and the intent disagree they will go with the letter if it’s more convenient for them.

  - Swarms of agents black boxing goals like “[<u>optimize my ad spend</u>](https://stratechery.com/2025/claude-4-anthropic-agents-human-ai-agents/)” will lead to bizarre grotesque results.

- The same origin policy is not a natural law.

  - It is a *human* law.

  - It was a historical accident!

    - I learned this week that apparently it wasn’t even well considered.

    - It was a hotfix the Netscape team decided on one night to handle an early Javascript security error.

    - Apparently Tim Berners Lee doesn’t like the policy.

  - It started as a quick fix, and then because it was a reasonable simplifying policy it grew and grew in momentum and import until now it feels like a law of gravity.

  - But unlike a law of gravity, it can be changed.

  - We made it, we can change it.

- The same origin model is a historical accident that was already showing its age.

  - Now in the age of AI it is clearly past its breaking point.

  - The age of AI needs integration, not isolation.

- Security models work inductively if a more savvy entity that the user trusts vouches for it.

  - When you’re unsure, you ask your more tech-savvy friend, “Would *you* trust this system with this use case?”

  - They in turn might ask their more savvy friend.

  - And on and on down to the person who audited the code herself.

  - If it’s just people you know, this comfort can take time to defuse through the network.

  - The friend need not be someone you know, just someone whose credibility you trust.

  - If a credible main-stream publication publishes a piece about why the system is trustworthy, that has a massive instant impact across the ecosystem.

- Someone pointed me this week at [<u>Admins, mods, and benevolent dictators for life: The implicit feudalism of online communities</u>](https://journals.sagepub.com/doi/10.1177/1461444820986553) by Nathan Schneider.

  - Sadly I don’t have access, but it sounds up my alley!

- Great piece from Matt Webb on "[<u>Multiplayer AI chat and conversational turn-taking: sharing what we learnt</u>](https://interconnected.org/home/2025/05/23/turntaking)"

- A haunting signpost: an [<u>AI-generated short film reflecting on the inner lives of AI actors</u>](https://x.com/hashemghaili/status/1927467022213869975?s=51&t=vzxMKR4cS0gSwwdp_gsNCA).

  - Like a Black Mirror episode!

- Decentralization has significant coordination costs.

  - Coordination costs scale super-linearly.

  - The benefits of decentralization are abstract for most people.

    - They’re more about downside capping of tail risk.

  - That’s one of the reasons that things like convenience and innovation-rate often win out in practice.

- The openness of a system is entirely down to whether there is a single asymmetrically powerful participant.

  - An “open” ecosystem with a massive single player can change the behavior of the system at will.

  - The standard only has power if it has a long streak of being respected (making it more shameful to break precedent) or there’s a rough balance of power in implementors.

  - That’s why the definition of how open a system is not tied to the license of the IP or whether there’s a standards body.

  - It’s defined entirely by how hard it would be for the ecosystem to recover if the most important entity went evil, greedy, incompetent, or lazy.

- Decentralizable systems are more credible if there’s a published roadmap.

  - Many systems aspire to be increasingly decentralized over time.

  - But decentralization has a cost; it slows the rate of innovation, and trades it off for the possibility of ubiquity.

  - But if the system is not yet good enough to become ubiquitous, then as innovation slows it can only hit its asymptote, because it can’t compete with other alternatives as effectively.

  - There will never be a good time to decentralize more, especially if the creator has to choose to cede control (vs it happening naturally as the investment of other entities ramps up).

  - So if there’s some central piece of control the creator has to delegate, it’s better not to have it be an all-or-nothing moment, because the creator might delay indefinitely.

  - It’s better if there’s a published, smooth roadmap of milestones and things that should happen when those milestones are hit.

  - If the creator doesn’t actually do what the roadmap says at those milestones, it reveals that their word shouldn’t be trusted, which would lead the ecosystem to lose momentum.

  - That danger forces the creator to behave aligned with decentralization, even if they might later not want to.

  - It’s similar to throwing your steering wheel out the window to win a game of chicken.

- Just because a thing is lindy doesn't mean it wasn't originally an accident.

  - Sometimes accidents that stick around just so happen to have been lucky.

  - The reason they stick around is because they were lucky, not necessarily because the creator knew what they were doing.

  - The things that weren’t viable faded away and we never talk about them again.

  - The things that happened to be viable stick around and thus they’re more likely to be a thing people remark on.

- CRDTs every so often have weird merge issues.

  - They’re eventually consistent… but not necessarily to a semantically *coherent* state.

  - These little errors are often not a big deal on their own.

  - If a human is watching, they can correct the error before it does much damage.

  - But if no human is watching, they tend to accrete on top of each other.

  - Each error has a super-linear rate of oddities.

  - Systems without humans in the loop on a continuous basis (e.g. background logs processes) are not viable with CRDTs or other systems that tend to accumulate errors at continuous rates.

- It’s easy to tell if a limit in a computer system was set by someone with a technical background or someone without one.

  - Engineers always pick a power of two (or one lower, if it starts at zero).

  - A secret hint about the creation of the system that is only obvious to people with a technical background.

- A codebase is not random.

  - It’s the accumulation of millions of intentional decisions by humans.

  - So wizened engineers can sense whole histories just by glancing at a codebase.

  - Codebases are not just technical artifacts; they are *socio*techno artifacts.

  - By understanding there is a sociological dimension, you can understand any given codebase on a much deeper level.

- “Should the floating castle we’re designing have a pool or not?”

  - “I can’t get past the floating castle part, because that's impossible!”

  - Someone this week was trying to get me to opine on how payments in an agent to agent ecosystem would work, but I was having a hard time.

  - Prompt injection and the laws of physics of the same origin paradigm (where code can have irreversible consequences, e.g. a network request that exfiltrates information) make the agent to agent ecosystem as currently envisioned obviously impossible to me.

  - As a result I can’t make any other coherent predictions within that future people imagine.

- The more powerful you are the more shielded you are from the consequences of your actions.

- The Saruman magic is an emergent social imaginary.

  - A shared reality distortion field, that when powerful enough, can put an actual dent in the universe.

  - Powerful but unstable, potentially supercritical.

  - It’s the emperor has no clothes kind of system.

  - Can be collapsed in an instant with one child laughing.

- If you’re only thinking in one dimension you’ll waste tons of time down dead ends.

  - They'll look like non-dead ends, but are dead ends in dimensions you can't even see.

- Taking notes during a conversation for me is like chewing on thoughts to start digesting them.

  - If you’ve met me in person you know that I have a habit of writing down notes live on my phone in a conversation.

    - I know it can be disconcerting for the other person, sorry!

  - But if I don’t capture them in the moment and context, they’ll be harder for me to extract later, even if I had a transcript of them.

  - It’s much harder to digest insights after the fact if they haven’t already been a bit predigested already.

- If you’re doing something new, it’s inherently hard to explain to others.

  - You don't have the touchpoints of "this thing you already are familiar with, but with this small tweak."

  - The minimal viable explanation has more steps in it.

  - Each step leads to a super-linear degradation of likelihood being received.

    - At each step there is some likelihood a listener gives up.

    - Over multiple steps, that compounds.