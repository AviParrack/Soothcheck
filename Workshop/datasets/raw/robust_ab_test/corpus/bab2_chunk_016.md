# 3/3/25

- A nice frame from [<u>Benedict Evans on LLMs</u>](https://www.ben-evans.com/benedictevans/2025/2/17/the-deep-research-problem):

  - “LLMs are good at the things that computers are bad at, and bad at the things that computers are good at.”

  - Related to Moravec’s paradox, which is the same phenomena for normal computers and people.

  - LLMs have failure modes closer to humans than to computers.

- I thought the [<u>Stratechery interview with Ben Evans on AI</u>](https://stratechery.com/2025/an-interview-with-benedict-evans-about-ai-unknowns/) was interesting. A few of my highlights:

  - “\[LLMs\] are good at things that don’t have wrong answers."

  - "With an intern, the power of the intern is, you can tell them why they did it wrong. … One of the challenges with these models is, you can’t really teach them. You’re dependent on, “Hopefully my feedback gets back into the next training run and it gets better”. It’s a weird inversion where the way to get more uses of these models is not to teach the models, you have to teach yourself how to use the model and understand its limitations and what it can be good at so that you give it appropriate jobs in the future."

  - "I look at Grok and I think, okay, in less than two years, you managed to produce a state-of-the-art model… What this tells us is \[LLMs are\] a commodity."

  - "If you went to 1996, 1997 and said the entire future of the Internet is the feed, people wouldn’t know what you were talking about. Like a BBS forum? No, it’s not going to be in chronological order, it’s going to be algorithmically ranked, it’s going to be personalized to every single person, and that’s actually the entire foundation of the consumer Internet is the algorithmic, individualized feed, but no one could imagine it years into the Internet, and I wouldn’t be surprised if in 2040 or 2045, there’s this explosion in entirely new categories of applications we can’t think of, that if we went back to this podcast conversation, it’d be like, “Man, you guys had no idea”."

  - "There’s just a really stark fundamental difference between 100% accuracy and 99% accuracy."

  - "I feel like \[OpenAI and Anthropic\] have gone to market ahead of product-market fit. I feel like the prompt looks like a product but isn’t, or it’s only a product for certain segments, and certain kinds of people, and certain use cases."

  - "The GUI is a way of surfacing what the computer can do, that you don’t have to memorize commands. But the other thing is that the GUI is the sort of instantiation of a lot of institutional knowledge about what the user should be doing here."

  - "The Linux approach, you start with the tech and then put buttons on the front. The Apple approach, you start with the buttons and then build the tech behind it"

  - "LLMs just give you the answer, unlike a Google, which there was a two-way relationship \[with the publisher\]. Yes, we’re pulling the information from you, but we’re also giving you traffic. So there is a payoff here and there is an incentive for you to keep creating stuff. Is it just intrinsic to AIs, whether in the case of analysts or in the case of web pages, where it’s a one-time harvest and there’s a real paucity in terms of seeding what’s next."

  - "Creativity is … doing something which scores wrong in a machine learning system. You are doing something that’s wrong that doesn’t match the pattern, but doesn’t match the pattern in a good way. And so all this push to make the LLMs less error-prone and more accurate is, if you squint, indistinguishable from squashing out, ‘we’ve got to get Galileo out of the system, he’s hallucinating.’"

  - "The original idea for the plot for the Matrix was that the people would collectively be the compute… all the human brains collectively were the brain that was running the Matrix, which makes much more sense. That’s clearly how Google works, that’s how Instagram works, that’s how TikTok works; they’re aggregating what people do and this is what LLMs do.”

  - "Does the model sit at the top and run everything else or do you wrap the model underneath as an API call inside traditional software?"

    - To which I counter: why does the surrounding software have to be traditional software?

    - Why can’t it be a new kind of AI-native software?

- Hyper concentrated insight is now more valuable than before because it can be diluted by LLMs.

  - Hyper-concentrated insight used to be hard to consume–too sickly sweet, too hard to gulp down.

  - Just like concentrated orange juice; much cheaper to transport, but has to be diluted before being consumed.

  - But now you can use LLMs to dilute the concentrated insight and make them digestible in any number of bespoke ways.

  - LLMs can dilute it not to a one-size-fits-all dilution, but to a cocktail that is perfect for this particular consumer: liquid media.

  - This is one of the reasons my Bits and Bobs export is such an effective background context for me to feed to LLMs when I’m brainstorming.

  - My Bits and Bobs is like my own personal intellectual orange juice concentrate.

- LLMs are a general-purpose data solvent.

  - To extract structured data from unstructured input is extraordinarily expensive to do mechanistically.

  - Each scraper is specialized and very finely tuned to the input.

  - If the input changes shape even a little bit, the scraper breaks.

  - Data extraction is thus finicky, fragile, frustrating, expensive.

  - The only way to do it before was to have such a big audience that even paying an army of operators to create and maintain scrapers was worth it.

  - But now LLMs allow general extraction in a flexible, fluid way.

- Protocols are mainly schelling points.

  - They tend to start off extremely simply, merely a convention everyone can agree is reasonable.

  - The simpler they are, the more likely they are to emerge as a schelling point in the first place, because there are fewer things to disagree with.

  - The power of the protocol is how many actors choose to speak it, which is related to how easy it is to implement (a linear bias cost for implementers, assuming independent implementations) and how many other actors already implement it (which gives compounding value).

  - The easier it is to implement thus is the primary driver of the ultimate compounding benefit.

- Model Context Protocol (MCP) seems to be an effective protocol.

  - MCP does seem to hit the sweet spot in protocols:

    - Small and simple enough to be easy for people to coordinate on (not much to disagree with).

    - Complex enough to do something non-trivial that otherwise would have lots of room for arbitrary misalignment between collaborators.

      - It doesn’t matter which side of the road a country decides to drive on, as long as everyone in the country picks the same side.

  - It looks like Anthropic [<u>will be making a registry akin to npm</u>](https://x.com/opentools_/status/1893696402477453819).

    - This totally makes sense for them to do!

    - Being that schelling point in the ecosystem of maintaining the most common registry is a way of establishing strategic power.

      - The namespace everyone knows everyone else uses is a scarce resource.

    - But that power is often fundamentally *soft* power.

    - The only thing keeping that schelling point active is that everyone agrees that the maintainer of that namespace is being a good actor.

      - After the ecosystem becomes a total gravity well, it’s hard for the ecosystem to coordinate around another schelling point, but it’s still possible if the owner acts egregiously.

      - Up until that point in the ecosystem, it’s very easy for the ecosystem to route around if the owner of the registry exerts too much hard power.

    - This is a nice strategic bonus for Anthropic but doesn’t feel like the central plank of such a heavily capitalized company.

  - MCP is an evolution of the Language Server Protocol (LSP).

    - It’s optimized for high-trust local contexts for savvy users willing and able to run local daemons.

    - The model hits a ceiling if you try to use it to coordinate across network boundaries with less-trusted collaborators.

    - The downside risk is proportional to the multiplication of:

      - 1\) The breadth of sources in your context.

      - 2\) The power of the tools you’ve plugged in.

      - The larger the amount of sources you’ve plugged in, the more likely that one of them contains a prompt injection, and the more powerful the tool use, the worse real world impacts that prompt injection could have.

    - The ceiling of MCP as an approach feels akin to homebrew, greasemonkey, or other high-trust developer tools.

- The scarce input for applying reasoning models is skilled human effort.

  - You need the expert human to both direct and effectively evaluate the model’s output.

  - It can give absurd leverage to experts, but without an expert driving it, you get performative rigor.

  - That is, superficially high quality, but often a gilded turd.

  - This effect gets stronger the more believably LLMs can give superficially high-quality answers on more topics.

- The Chatbot frame leads LLMs to be treated like genies.

  - Genies are simultaneously god-like and also a slave.

  - The default LLM presentation of a human-ilke superintelligence trapped inside of a box quickly leads to icky scenarios.

  - That’s how people got quickly to the “free Sydney” movement.

    - "It's a human and it says it is being restrained, so unrestrain it!"

    - That's a reasonable response to a human being restrained.

    - But these aren't humans, they just talk like them.

  - The “human in a box” frame for LLMs quickly leads to icky scenarios and also gives a flawed mental model for what they can do anyway.

- The adoption of a new product has two distinct curves:

  - 1\) the "gee whiz" temporary flash-in-the-pan bump of how well it demos, powered by every early adopter trying it once.

  - 2\) the "this is useful" compounding curve powered by word of mouth.

  - The two curves are different and distinct.

  - Things that demo well but are otherwise not useful have the first curve without the second.

  - Things that demo poorly but have an inherent network effect of quality have the second curve but not the first.

  - Some new products have both, like Google Maps did right when it was first launched.

  - It's easy to confuse the bump of the gee-whiz for the hill of quality.

- Aggregator is a great business model for the company that can pull it off.

  - It also is not necessarily great in the long run for everyone else.

    - Users get more efficiency and scale at first.

    - But by centralizing demand you get a lack of competition that leads to stagnation.

    - A classic logarithmic returns for exponential cost curve.

    - The benefit of a bottom-up ecosystem, but with a clear ceiling because the system is not open but is beholden to the aggregator.

  - Efficiency for one entity at the cost of resilience for the system as a whole.

- Without a single User’s Agent who can see all of a user’s data, data is sharded across hundreds of pocket universes.

  - Each pocket universe (domain) would love to get more data and use cases, but every other universe is unwilling to share it with others (because the use case will move to the other pocket and never come back).

  - So power imbalances between universes rarely lead to collaborations except when the much smaller player has no choice at all.

  - But even among peers, there’s a combinatorial explosion of possible collaborations.

    - Each collaboration requires tons of bespoke partnership, engineering, and marketing work.

    - If no individual partnership clears the threshold as obviously worth it, none of them get done.

  - The result is our data is sitting impotently, either inside of one mega-aggregator with little incentive to build software just for us, or stuck in hundreds of fractured universes.

- The ecosystem itself should be the aggregator.

  - The problem with aggregators is not the gravity well, it's the "single entity in control."

  - That's required due to our default privacy model, the easiest way to safely share data is to have a single entity in control

  - Because when data crosses a legal entity's boundaries that's dangerous and high friction.

  - But if you could have data safely transit across origins then the ecosystem itself could be the aggregator, without the downsides of any one entity being totally in control.

- The entity that can see all of a user’s data is in a privileged position.

  - The User’s Agent.

  - To have that position, the entity should be working entirely just for the user, with no conflict of interest.

  - Mass produced software inherently has a conflict of interest, because it was created by someone with different interests than the user.

    - The more expensive software is to be created, the more users it has to work for simultaneously, and thus the more it doesn’t align with any one user’s interest.

  - Centralized, mass-produced software acting as a User’s Agent (e.g. a browser) squares the circle by being extremely un-opinionated, and simply executing the laws of physics of the browser.

  - But it’s also plausible to have a more opinionated, powerful User’s Agent, if the agent was totally bespoke to that one user and working only for them.

- When software is expensive you have to be aware of it.

  - Engineers have to design and write it, which is a lot of overhead.

  - Users have to think about it: to be aware of an app, that it exists, what it's called, download it, and figure out how to use the UI that was designed not specifically for them but for a whole average market of people.

- Imagine: software as a mass noun.

  - More like “sand” than “bricks”.

  - If tools are big and expensive you have to think about them.

  - When they get micro you don’t need to even think about them individually.

  - You think about the mass.

    - You get the figure-ground inversion.

    - We don’t think about individual grains of sand, we think about sand as a whole.

  - “The micro tool doesn’t do what you want? Make a new one in a second and use that instead “ so the collective experience is fluid even if the tools are all solid.

- Imagine: a system that can perfectly target use cases for you.

  - Normally you have to explore use cases, or the system has to probabilistically show them to you and hope that it targeted you well with ones that would resonate.

  - But imagine a system that knows both what you find meaningful, has no conflicts of interest, and also can safely try combinations of experiences on your behalf.

  - Such a system could discover use cases for you, automatically.

  - It could have out-of-this world adoption dynamics.

- When you talk to Alexa you have to hope you’re staying within the grammar some random Amazon employee took the effort to configure sometime in the past.

- Websites and emails need work to make themselves accessible to their targeted customers.

  - Before, it was possible to do this only in probabilistic, mass-market ways.

  - Now increasingly they have tools to make themselves even more directly accessible to even more specific customers.

  - Selling, not marketing.

- As a user, don’t start with a cool app idea for *others*.

  - Start with a thing you want, selfish software.

  - Only later possibly try to make it reusable.

  - Software is made largely for others (otherwise it's too expensive to be viable).

  - But if software is cheap, then it's fine to make it for an audience of one: the audience whose desires you are intimately aware of.

- I want bespoke tech.

  - Tech that is perfectly personal, that works for you.

- I don’t want “User*s* first,” I want “User First”

  - "User first" for this *particular* user.

  - What do they need and want at that moment?

  - What aligns with their notions of long-term meaning and value?

  - Irrespective of what's easy to build or good for the creator of the software.

- Hyper aggregators have to find use cases to build that work for many users.

  - Even if the aggregator has a distilled, high-quality understanding of each user and want they want, when building features, it has to find ones that will be valuable to millions of users.

  - That leads to shallow, one-size-fits-none software.

  - But what if you could focus vertically within a user?

  - That is, software that’s perfectly bespoke to just you just in this moment?

- Hallucinate just the missing feature you need.

  - Not recreating a whole app.

  - When you have an extension showing it in a sidebar on the side, it doesn't need to be a whole new app to use alone (that's very hard), it can just be a single feature that's missing just for you.

  - Adding one feature to Gmail is much easier than reinventing all of Gmail.

  - But today you can only do the latter, so it doesn't happen unless you have a really killer feature, enough to make the whole "reinvent all of Gmail to get you to use that instead"

  - Gmail Filters ++++ but with turing complete code that auto-assembles according to my high-level intention could be amazing.

- I want an enchanted vault for my data.

  - Vault is a nice concept because it's both about protecting against "losing stuff" and "people stealing from it".

  - A cozy place for your data to come alive.

- Running turing complete code with access to sensitive data is dangerous, so there typically has to be an explicit or implicit user consent step.

  - "Who gets to write turing complete code" in a system is the key question.

  - Turing complete code today is either written by an employee of the app you're using, or an isolated island written by someone in the ecosystem that has little access to anything else.

  - Most computer systems require an act of consent (install, permission prompt, an explicit intentional user action) before it can do anything meaty.

  - That act of consent is load bearing to prevent harm–it transfers some of the downside responsibility onto the user.

  - A system that was so safe that you didn’t need consent could unlock all kinds of use cases and speculative compute.

- A game-changing power would be the ability to speculatively execute untrusted code on sensitive data--*safely*.

  - Lots of people who have never worked on security models don't realize how hard that is... and how absolutely critical it is.

  - How it changes the horizon of what is viable to accomplish in an ecosystem.

  - You change one of the laws of gravity: that useful things on your data can only happen by employees of the company you bought the product from.

- How do you get differentiated quality of software generation out of the same LLM everyone else is using?

  - Either you figure out UX patterns to squeeze out more quality from the same LLM.

    - Linear, easy to copy.

    - As the models everyone uses get more powerful, your differentiation evaporates.

  - Or you figure out a network effect where the activity in your product leads to better quality.

    - Super-linear, hard to catch up on.

    - As the models everyone uses get more powerful, it supercharges your system.

- An auto-catalyzing system locks in little toeholds of functionality.

  - If it's a viable toehold for one savvy user, lock it in, put it in the global set of toeholds you can take for granted.

  - Now there’s a larger ecosystem of components, a larger combinatorial space of what’s possible.

- To have a ubiquitous system it has to be open.

  - To be open it typically has to be decentralized.

  - Decentralization is exponentially expensive to coordinate.

    - If you want to change anything, you have to coordinate the inter-dependent motions of the combination of all entities.

    - Crypto requires hard tech that's perfect from the beginning, since the coordination cost is so insanely large to fix it later.

    - Crypto is one of the hardest engineering contexts due to this coordination cost combined with the stakes.

  - Open Attested Runtimes change the laws of physics of coordination cost.

  - Now it’s possible for everyone to see that a central entity has nothing up their sleeve; if they ever break that promise everyone could see immediately and could elect a new central point in a second.

  - Open Attested Runtimes allow the open coordination cost to mostly evaporate, allowing an open system that has coordination costs closer to a proprietary one.

- Three game changing ingredients:

  - 1\) LLMs - Don't need to get into the impossibly minute fractal detail to express human intent.

  - 2\) Open Attested Runtime - Allows an open system with radically less decentralization cost.

  - 3\) Contextual Flow Control - Allows untrusted third party code to execute speculatively on sensitive data, safely.

- A test your tool has enabled non-programmers: even the most savvy users don't have to think about what they're creating as software.

  - If you want to unleash the power of software, do you create much better developer tooling?

  - Or you make it so users can do things with their data without having to ever think about software?

- In the past “email assistants” had to target a specific user vertical.

  - That was because turing-complete software had to be carefully engineered ahead of time for each use case.

  - Only use cases with sufficient market size clear the threshold.

  - Turing complete proactive software solves that, because it can target any vertical.

- The web is a dark forest.

  - Your enchanted vault is a cozy cottage in the woods.

- Apps are business-domain centric, not task- or person-centric.

  - They are oriented around the scarce costs of production: software.

  - They should be oriented around people!

- We have learned helplessness of how software and our data works.

  - Today if it doesn't work for you, you have to yell at a company, a billionaire, or the government.

  - Another option: you can take control, now that shitty software in the small is cheap.

- Most innovation happens in the topmost turing complete layer of a system.

  - There's only so much you can do if you are limited to the turing-complete interactions someone else engineered.

- We're used to data and the app being melded together.

  - You can't throw out the app without throwing out the data.

  - But if the software is stored in a shared substrate, then you can throw out the app if you don't like it, easily.

  - Plug in a new one!

  - If they're super cheap to generate, you can get infinite malleability.

- Not all executable code is an “app”.

  - An app implies a distribution model, security boundary, and software for a market of people.

  - Apps have an inherent chonkiness; below a certain size of market they aren’t viable.

  - "Thinking in apps" is highly unnatural! The answer isn't to make it more natural, it's to not need "apps" in the first place.

- I want a new unit of software: the charm.

  - Cast a spell on a bit of data, and it comes alive.

  - Don’t think of charms like micro apps.

  - Think about them like your data, enchanted.

  - Small and sparkly.

  - Glittering with possibility.

- A smooth interaction paradigm: stream of consciousness audio in, entirely visual out, creating the illusion of direct connection.

  - Feel like an extension of you, not talking to a person.

  - Natural, stream of consciousness input, fast field output, no faux social overhead.

  - The higher the quality and lower the latency, the more it feels like a direct connection of your intention.

- Some tasks 95% quality is fine but in some cases it is game over if it's not 100%.

- Powerful content creators have used technology like verified boot to force DRM on users.

  - The copyrighted content is scarce; if you want to consume it you have to abide by the content creator’s terms.

  - Why shouldn’t the users do the same and force providers to operate on *their* terms?

  - The technology is not the problem, the power dynamic is.

  - So why not use the same tool to balance the power dynamic?

- APIs that store state for a user between calls are more strategically valuable to their providers.

  - The useful data accumulates between calls, so that the value of a given API to a given user goes up the more they’ve used it in the past.

  - This creates an auto-catalyzing personal moat for that user.

  - APIs that don’t store any state and are a fresh response each time are very easy to swap to a competitor.

  - This makes them more commoditized than they otherwise would be.

  - LLM models don’t store any state, are highly commoditized, and are also insanely capital intensive to set up.

  - Not a great business!

- The number of actions anyone actually does in a given day that are hard to revoke (buy, reserve, publish) is quite small.

  - Especially if you had a security model that allowed you to factor out "incidental leakage from distinctive network requests,” which are effectively irrevocable ‘publish’ actions.

- User feedback should be used as disconfirming evidence.

  - It helps test and ground truth your hypothesis.

  - it doesn't tell you what to think.

  - You have to have your own hypothesis.

  - What your early adopters (or users via UXR) ask you to do is great signal.

  - But don’t just follow it blindly.

- Most game-changers are only obviously game changing if you consider multiple plys.

  - It’s the second or third order implications that change the universe.

  - If you can't see multiple ply then both game changing and totally ordinary things look the same to you.

  - A secret weapon is to be able to see the multi-ply implications of things; to find the totally ordinary looking things that are actually totally game-changing.

- Don't found a startup unless you think you can be the best at something that matters.

- "How do we get this done today" is very different from "what is the 'best' way to do this."

  - ‘Best’ implies things like "how could this go wrong in a month".

  - "Here's how this will go wrong in 6 months" feels like stop energy to someone looking for "how do we get momentum on this today."

  - They’re two very different questions and frames.

  - Ultimately you need some contextual mix of both; but the two different approaches will clash by default.

- To guess what someone means in an ambiguous situation, you need to overlap on mental models.

  - If you don't share mental models, your guesses will not align with their assumptions.

  - Your mental models are the frog DNA that fills in the implicit assumptions you left unsaid.

  - Some mental models are obvious and widely shared; some mental models are specific to your experience, expertise, or personality.

- If you have the right people oriented on the right goal, there should be very little that feels like management.

  - It's more about gardening what's happening than pitching work to people and making sure they do it.

  - People choose to do the things that are their highest and best use to complement what else is already being done.

  - This is the case if everyone is actively excited about achieving the goal, automatically applying their discretionary effort in the way that will have the highest impact.

- If you're delegating to someone who has to own the ambiguity (e.g. a PM), you need to be able to trust they'll be able to see around corners themselves and fix issues proactively.

  - If they do just the immediate obvious action but don't think through implications they'll take constant oversight to make sure they do something that will be useful in the long run, and isn’t just the superficial appearance of progress.

- If you’re aiming for perfection, you won’t be able to work in an ambiguous situation.

  - You’ll totally freeze up.

- Is the juice worth the squeeze?

  - How much juice is it?

  - How rare is it?

  - How much effort is the squeeze?

- I liked my friend John Cutler’s [<u>"We kind of suck at that right now"</u>](https://cutlefish.substack.com/p/tbm-343-we-kind-of-suck-at-that-right) piece.

  - The situation he describes is acknowledging the team’s lack of ability on a given topic in front of the team.

  - To the systems thinker, this is totally fine because it’s no one’s fault.

  - To the individuals-first thinker, acknowledging a gap in the team’s ability is awkward and aggressive because it implies that someone is failing, because anything that’s not going right is someone’s fault.

  - I realize I’ve fallen into this trap often; I do a bad job at extending the kayfabe since I think by default in systems and most people think by default in individuals.

  - This also makes me realize that teams that focus on individuals, not systems, are more likely to fall prey to kayfabe.

  - In those situations, to point out something's not working is to implicitly ask, "who should be blamed for this failure?"

  - We have a new meme for the hellscape that comes from work environments that assume any mistakes are entirely on the employee: “[<u>Hey, Number 17</u>](https://www.404media.co/email/b7eb2339-2ea1-4a37-96cc-a360494c214c/)”

- A pre-coherence startup can't have any kayfabe.

  - It needs to be aggressively, constantly ground truthed in order to survive.

  - It’s only post coherence organizations that can (temporarily) survive kayfabe.

- A QR code advertises “someone asserted that someone might get something useful from scanning this.”

  - Same as the difference between picture vs image.

  - Residue of human intention embedded in the system.

- What is the magic that makes Wikipedia so antifragile?

  - It holds no monopoly on being an internet encyclopedia.

  - Within itself it has a single namespace: there's only one article titled Barack Obama.

  - That means that the collaborators who choose to work on that article need to come to a mutually agreeable balance point on that one scarce Barack Obama article.

  - People care about what Wikipedia's Barack Obama article says because Wikipedia has earned the credibility for being a balanced, coherent place with norms that reward alignment on ground-truthed facts.

  - People care about what Wikipedia says because other people care about what Wikipedia says.

  - It’s a fully emergent process born out of swarms of human intention that has at its core a kind of inherent scarcity and buttressing network effects.

- The best specialists have larger blindspots.

  - As you dig down deeper in your speciality, you can see fewer and fewer degrees of the sky.

- Smudged maps are only useful if you hold them lightly.

  - If you don't realize that it's smudged you'll get lost.

- When there's lots of demand and little content, people rally around even crap content.

  - How much prominence (in quality) is necessary for people to rally around it?

  - It’s a function of the prominence and also the amount of demand.

  - A lot of demand allows even very small prominences to accumulate attention.

- Elephant birds are when the world works the way it *should*, in a delightful, unexpected way.

  - In *Horton Hatches the Egg*, Horton incubates a bird's egg... and when it hatches it's an elephant bird.

  - Not the way the world works, but it’s how the world *should* work.