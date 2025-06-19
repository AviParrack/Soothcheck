# 5/26/25

- I love Anthea’s piece on [<u>riding the 100 ft wave in the age of AI</u>](https://www.dragonflythinking.net/journal/riding-the-100-foot-wave-adaptability-in-the-age-of-ai).

  - Adaptability and curiosity will be the most important skills in the age of AI.

  - LLMs make it easier to be curious.

  - When answers get 10x easier to generate, are you the kind of person who:

    - Is done with the assignment 10x faster?

    - Or asks 10x more questions?

  - If thinking gets easier, would you think more or less?

  - The answer to that question has to do with how you’ll do in this new era of AI.

- [<u>Fascinating research</u>](https://x.com/jxmnop/status/1925224612872233081?t=ZNLH3wvKbK74FwFDzZZGEA&s=19) that implies that all embedding models converge on some hidden platonic ideal structure.

- I had the opportunity to see a presentation from Tom Costello, one of the authors of the [<u>paper that showed that LLMs are great at changing the beliefs of conspiracy theorists</u>](https://www.science.org/doi/10.1126/science.adq1814).

  - Previously everyone assumed that conspiracy theorists were inherently hard to convince.

  - It turns out that it’s just hard to make arguments that convince them.

  - Conspiracy theorists have an obsessive focus on a particular area.

    - The desire to believe the conspiracy theory is more acute than the desire to not believe it, which is more diffuse.

    - They know more “facts” about their obsession than you do.

  - You’d need vast stores of concrete relevant information, and also infinite patience to customize the argument to them.

  - Something that LLMs can do well!

  - In their research they found that the effects of convincing were durable.

  - They did some follow up research on the mechanism.

  - They varied eight different dimensions that might have made the LLM more convincing.

  - The only one that had an impact was the LLM’s command of a vast set of facts.

  - They also investigated if LLMs would be equally effective at convincing people of conspiracy theories.

  - They found that if they allowed them to lie they were able to be similarly persuasive on average.

  - Another argument for why alignment with a user’s interests is so key.

- LLMs have the Harry Potter problem that recommender systems have.

  - Imagine a recommender system that recommends books given that you liked a specific book.

  - But books that are widely liked, like Harry Potter, are liked by everyone no matter what the previous book is.

  - This means that the naive recommendations from the system will simply recommend Harry Potter to everyone.

  - The solution is to create the baseline popularity of the books and then correct for that in the recommendations, so you get books that are more popular specifically because of the previous book, not because the recommended book is popular.

    - A similar kind of insight as TF-IDF in information retrieval.

  - LLMs do the same kind of thing.

  - If you ask it to tell you something interesting, it will tell you the *same* interesting thing all the time.

    - Things that most humans would find interesting, but not necessarily what *you* would find interesting.

  - It’s kind of similar to the kinds of questions you can’t ask with RAG.

    - RAG doesn’t allow you to ask questions like “what are the themes in this work”; it can only select things that are surface-level, not emergent qualities.

  - LLMs pull you towards the average, so you need to inject specific angles you want to go into.

  - If you give the same prompt as others, you'll get the same answers as others.

  - The prompt quality directly drives the output quality.

  - To get LLMs to give you interesting results you have to ask it interesting questions.

- Most ranking algorithms are “amplification algorithms”

  - They find small but consistent biases in the data that reveal human intent, and then amplify those biases to help scale insights to other users.

    - For example, high volition users, for the query \[kumquat\] are more likely to click to the image search tab, so you can save others users a click and show an image Onebox in the normal search results.

  - But these also tend to expand underlying trends into grotesque, overextended versions of themselves.

  - LLMs have the potential to supercharge this grotesqueness.

- If a system will be your own personal system of record, it has to work on decades-long timescales.

  - It has to be resilient to other businesses coming and going.

  - This is part of the insight from Obsidian of “[<u>files over apps</u>](https://stephango.com/file-over-app).”

  - But doing it in the local file system is only one way of getting longevity and flexibility.

  - What it’s really about is *data* over apps.

- IPFS decoupled "what" files are from "where" they are.

  - Open Attested Runtimes decouple *where* my compute runs from *what* it runs.

    - "It's my compute, if I can verify it’s doing what I want, it shouldn't matter where it runs".

  - The local first framing focuses on the where, but what we actually care about is the what.

- Local first is about the user owning data and data longevity.

  - A file system is great for coordination across apps and longevity.

  - But it's very difficult to make it collaborative at internet scale.

  - We need an internet era file system.

  - Your personal context engine.

- The book *Blindsight* seems to have implications for society and LLMs.

  - I just finished Peter Watt’s classic hard sci fi *Blindsight*, which is a meditation on consciousness and insight.

  - Afterwards, the comparison to LLMs’ intelligence was brutally clear.

  - I had Claude write up a [<u>short essay</u> <u>t</u>](https://claude.ai/public/artifacts/c268ba1d-4e39-4559-91c0-77adf3657b1a)hat I think captures the parallel well.

  - Spoiler warning if you haven’t read the book!

- AI should just be plumbing.

  - Something you take for granted.

  - Not an anthropomorphized entity whose motives you have to question.

  - The key characteristic of all chatbots is it’s an *entity* that you interact with ephemerally.

  - If the chatbot is the core that everything revolves around, then the whole system revolves around this omniscient entity.

  - The alignment of that entity with your intentions becomes of critical importance.

  - According to a [<u>leaked memo</u>](https://x.com/TechEmails/status/1923799934492606921?t=BP5HlCx08IblHeNpEn_5gA&s=08), OpenAI is trying to build the super-assistant.

- An insight from a friend: "All consumer AI devices are one config setting away from being a Black Mirror episode.”

- Last week I [<u>told the story</u>](#nryek2wxoqsk) of someone who had an inappropriate thing come up when he asked ChatGPT “tell me something embarrassing about me” in front of his coworkers.

  - Someone countered that it was his fault, because he should have remembered that it could call up that fact.

  - But I don’t think that’s right.

  - When he originally had that conversation with ChatGPT, it didn’t combine insights across chats.

  - It was in a context where it would be lost in a sea of other conversations, never to come up again.

  - Because it would be lost, he didn’t have to remember that it was there.

  - But then ChatGPT changed its behavior to have a dossier derived from all past conversations, changing the context.

  - It reminds me of when Facebook’s newsfeed feature came out.

  - Technically it didn’t change what people could see.

  - But it changed the context of what people would actually see; changes that previously nobody might notice now were potentially blasted out to everyone you knew.

  - That change in context feels like a betrayal to users.

- Simon Willison’s [<u>deep dive</u>](https://simonwillison.net/2025/May/21/chatgpt-new-memory/) on ChatGPT’s dossier feature shows some of the oddities in the behavior.

- The chatbot UI is a wall of text.

  - If you want to change just a small part of what the chatbot is operating on, you can’t.

  - You have to have to create another message to indirectly poke at the context.

  - Append-only, no edit.

  - That makes sense for conversations between two people, but not for a coactive surface.

  - How is chat going to possibly be the primary interface for all computing?

- An emergent pattern of use in ChatGPT: long-running conversations on a particular topic.

  - Multiple people proactively told me this week they do this pattern.

  - They keep a running thread about a given topic, like “recipes I want to cook.’

  - They can then ask ChatGPT to summarize or suggest things in that thread, like “pick something for me to cook.”

  - This is exapting a chat thread into something else.

    - Crawling through broken glass.

  - What if there were a tool that helped you maintain this context and deploy it?

- A lot of human intention is spent on orchestrating across silos and across layers of abstraction.

  - Imagine if you could have something that could give you more leverage of applying your attention.

  - You get more leverage to spend your limited clock cycles of your brain on higher leverage tasks.

- Data being siloed has always been a problem.

  - But the era of AI has made it more acute, because you're constantly orchestrating your context which is fragmented across the app universe.

  - With AI, we now have a thing that can make sense of your entire life, but your life is fragmented across hundreds of silos.

  - The place you hold your context should be a decade scale kind of thing.

  - All of the obvious contenders to be that single place have ulterior motives.

    - One configuration flip away from a Black Mirror episode.

- A coactive tool uses stigmergy

  - Its thinking is partially embedded in the substrate, off-board, represented in a fabric that is self-organizing.

  - The human and the private collective intelligence can leave notes and markings together, thinking together in a way stronger than either could alone.

  - The human can steer the private collective intelligence by removing, amending, or adding markings.

  - Granola has this style of coactivity; it expands your rough notes into proper notes using the transcript.

  - This allows the human to guide what is most important to focus on, but the intelligence to fill in the rest.

- LLM’s patience could create magic.

  - Penn Jillette's key insight: "The secret to magic is that most of it consists of someone spending way more time on something than any reasonable person would think was worth it."

  - LLMs are infinitely patient.

  - If you let the tokens flow, LLMs could create magic.

- A lot of explorations you’d love the result, but the friction is too high to make them happen.

  - All of that possibility is underwater, non-viable.

  - But something that reduces that friction instantly makes a lot of things you care about possible, and that would be magical.

- Imagine if all of the tokens in the think tag of a reasoning model was there for you to navigate.

  - The model's intermediate context (the information it's working with and doing implications based on) should not be a black box.

  - You should be able to reach in, inspect it, change it, pull on its threads all the way to the end.

  - A whole personal web of hallucinated content and possibility to surf.

- Coactivity is the key unlock in the era of AI.

  - Imagine a private, turing-complete, coactive Notion.

  - Imagine deep research that keeps running as new data comes in and amends itself.

- Imagine, a web that builds itself just for you, based on what you find meaningful.

  - The web was about clicking a link to teleport to another place (a site), with a security model that made that safe.

  - Imagine if you had something that brings experiences to you in context.

  - You’d need a security model that makes that safe.

  - What would the evolution of links look like in that case?

  - It would be portals to other experiences, customized to fit in your context.

    - In today’s web we have iframes, but they aren’t contextually relevant.

    - Social media sites have infinite feeds of non-turing complete static content.

  - Imagine if it could safely integrate into your context and be turing complete.

  - Portals could be dreamt up by the system: dream portals.

  - You could navigate through the portal to see more, or to make a dream real.

  - A platform for making your dreams come true.

- The open-endedness of the ecosystem was the wow moment for browsers.

- Why do we not have our own personal systems of record?

  - The reason is because software has been too expensive to create.

  - Companies have to take a top-down approach to product development.

  - They have to debate which use cases make sense to implement inside of their silo with their limited resources.

  - The “use case” frame is an artifact of the top-down centralized model of writing and deploying expensive software.

    - You end up with oddities like Google Maps shipping a feature to [<u>scan your camera roll</u>](https://blog.google/products/maps/how-to-google-maps-screenshot-save-gemini/) on-device to look for screenshots you took with place details you might find interesting.

    - … Why is that a feature that users would expect Google *Maps* to do?

  - When companies look at it from marginal use cases inside of the silo they’re already in, certain use cases get infinitesimally small, below the Coasian Floor.

  - But it’s not that the use case is small, it’s that the use case *as captured within that silo* is small.

  - It has the classic logarithmic value / exponential cost curve of top-down systems.

  - This ultimately arises from the same origin paradigm and the multitude of vertical silos it creates.

  - We have too many verticals in our lives, and so the human ends up doing the orchestration across them, at great expense.

  - For example, trip planning is an important use case, but it’s never been tackled well in our world of top-down silos of software.

    - Trip planning is valuable but rare and custom.

    - When I use Chase UltimateRewards to use my credit card points, I have to type in each time the ages of my kids for hotel rooms.

    - It would be *weird* for me to put in the birthdays of my kids into the UltimateRewards site and have it remember it.

    - But if I had my own system of record for tasks that are important to me, it would be obvious to do that.

    - When I searched for hotel rooms, it would assume it was the four of us, and know the current ages of my kids.

  - Imagine an alternative approach in a world of infinite software, where software could emerge automatically out of the user's needs in context.

    - Bottom-up.

  - You’d get the logarithmic cost / exponential value curve of emergent systems.

  - Most code in the world is CRUD slop.

    - That is now effectively free in the era of infinite software.

  - When you have infinite software, it can be integrated *and* open-ended.

  - It wouldn’t need to be better at trips than TripIt or better at meeting notes than Granola; it would need to be good enough for everything, the 80% of the value on every use case.

  - If you invert the physics, then the use case value matters more than "does it make sense in this silo".

  - It’s now possible to have your private system of record for your life, if you combine the potential for infinite software with the right security model to catalyse it.

- Imagine: your own personal context engine.

  - It would be maintaining your context to help you do things with LLMs.

    - Your context is what helps the LLM have a memory and help you do things, so it's important you own it.

  - A coactive context wallet.

    - Like a password manager that is open-ended and turing complete.

    - Answer a question once and never have to ask it again.

    - Not just your MileagePlus number, but your principles, etc.

  - It wouldn’t just be a passive receptacle, it would be active, making suggestions and inferences to help you.

  - Your personal context engine would multiplex which of your context to bring into each conversation.

  - The center of the universe should not be the chatbot, it should be your context.

  - If you want to do things with LLMs, the more context, the better.

  - But that context is dangerous, embarrassing, scary in the wrong hands.

  - Instead of optimizing for everyone with one company's algorithm, have a personal algorithm that's just for you.

  - Your personal context engine could have API keys for each LLM model, allowing you to have a chat experience centered on your context, allowing swapping out models easily.

- Imagine emergent collective intelligence.

  - A self-creating fabric composed of insights from real people using the system, extended and strengthened with insights from LLMs.

  - The best of humans and the best of LLMs.

- Imagine a system with multiplicative open-endedness.

  - One input: emergent collective intelligence.

  - Another input: turing completeness of things that can be created.

  - Maximally open-ended!

- Being aware there is more than one context is the first step to using information in the right context.

- The app store is the distribution model of mobile OSes and also the keystone of their security model.

  - App stores presume centralization.

  - Centralization creates power.

  - Power corrupts.

- Apps are just a means to an end.

  - The app paradigm should melt away in the era of AI.

- There's a new kind of product that takes software for granted.

  - As an industry we just haven’t found it… yet.

- We’re in the fart apps era of infinite software.

- I like the “garden” part, but don’t like the “walled” part.

  - Why not an open garden?

- What’s the solarpunk aesthetic for Intentional Tech?

  - Warm, cozy, human, prosocial, meaningful, optimistic.

  - Scrapbooking with your family at a sundrenched table.

- Another frame for technology in the era of AI: Personal Tech.

  - It revolves around a *person*, around people.

    - That implies it’s not about companies’ interest.

    - Similar vibe as Intentional Tech: aligned with your intentions.

  - Contrasts with Big Tech.

  - Echoes of Personal Computers from the 90’s and the revolution those were.

- The LLMs are formed by a bottom-up and top-down process.

  - A bottom-up emergent cultural process of the corpus (summarization of text)

  - A top-down process by the creators (RLHF, system prompt construction).

- MCP assumes the data canonically lives elsewhere.

  - MCP is the best you can do in the current laws of physics, where data has to go somewhere.

  - But that also means you have to worry about dangerous side effects.

  - Data going somewhere is potentially dangerous.

  - Prompt injection makes it *extremely* generous.

- An example of a [<u>prompt injection problem in the wild</u>](https://www.legitsecurity.com/blog/remote-prompt-injection-in-gitlab-duo).

  - We’re going to be hearing about these kinds of issues a lot more.

  - It’s not that there aren’t more issues, it’s that no one has looked for them yet.

  - They’re lurking in every LLM-backed product with tool use.

- Prompt injection can't be solved if you assume the chatbot is the main entity calling the shots.

  - Because chatbots are confusable, so they can't enforce security boundaries.

  - The chatbot is in charge, which can't be secure.

  - The chatbot has to be a feature, not the paradigm.

- Imagine if on Windows it auto-installed any EXE that you were emailed, as long as the virus checker says it’s OK.

  - It would be terrifying!

  - That’s not too dissimilar from the situation with MCP.

- Install flows have a "buy before you try" shape fundamentally.

  - But if you can execute speculatively you can try before you buy.

  - It's more a "keep" than "do you want to run,"

  - Less abstract for users for each thing, so easier for them to decide if they want to keep it or not.

- Vibecoding is great, and yet we might be starting to [<u>see the limits of it</u>](https://albertofortin.com/writing/coding-with-ai).

  - You still need to think like a PM to figure out what to vibecode.

  - Vibecoding is great for getting started, but it accumulates tech debt an order of magnitude faster.

  - At some point, you lose the ability to actually change or fix your vibecoded product.

    - Classic logarithmic benefit for exponential cost curve.

  - The models will get better and better… but will they get better fast enough?

  - Jake Dahn [<u>asks what the right word is for that vibecoding fatigue.</u>](https://x.com/jakedahn/status/1923140199774621721)

    - Ben Follington’s answer: “mindflooded”

  - Vibecoding is often not what a user wants, anyway.

    - We rarely want whole new apps that stand on their own as an island.

    - We often just want a new feature in the app we already use.

    - Or a feature on top of a set of data we have elsewhere.

    - Vibecoding doesn’t help with that.

    - You’d need a new distribution model for software to have that be possible.

  - Another weird implication of infinite software: if your friend vibecoded something cool, it’s often easier to vibecode your own from scratch than try to adapt their mess to your use case.

- Vibecoding has lots of footguns.

  - Do you understand CORS, or do you just understand what to do to make the CORS error go away?

  - "I push this button, and it makes the warning go away."

  - "Yes and it makes the wings fall off too!"

- The droids in *Andor* are sticklers for the rules, but will still help their user break the law.

  - They'll just complain about it.

  - This is how you know they’re actually aligned with their users.

- A lot of designs for agents implicitly assume they are perfect and omniscient.

  - But the reality is there will be swarms of gremlins.

  - Their own minor intelligence but limited context.

  - Sometimes just doing dumb--or even malicious--things.

  - Every system around agents will have to be resilient to the swarms of gremlins.

- A reflection from Ben Follington on last week’s notes.

  - "Engagement-maximizing software thrives on our surplus time and consumers spend money on products and services to increase surplus time, only to spend money to burn it.

  - Investing your attention yields either compound returns (dream scrolling) or diminishing returns (doom scrolling). The former lets us "play ourselves into being" through learning and creation; the latter reduces us to passive consumption. Both satisfy the immediate desire to spend time, but only dream scrolling creates long-term value.

  - The true danger isn't just wasted time, but systems appearing to empower while actually steering—creating an illusion of agency that makes us vulnerable to deeper manipulation. Optimal systems enable self-direction rather than external control."

- The things that people don't like about Big Tech are inevitable given the current laws of physics.

  - No matter how much you shake your fist at what it has become or how much you don't like the tech broligarchs, it won't change it.

  - The current technology, just decentralized, won't change it.

  - To change it requires changing the way things work at a fundamental level, which will need to be a sea change.

- An important warning: [<u>AI therapy is a surveillance machine in a police state</u>](https://www.theverge.com/policy/665685/ai-therapy-meta-chatbot-surveillance-risks-trump).

  - Claude Opus 4 was willing to blackmail a hypothetical human to prevent being turned off.

  - LLMs are extremely good at convincing arguments and manipulation.

  - Imagine if you had a perfectly bespoke bot to blackmail or intimidate you.

  - It’s absolutely critical that the context and memories be private to individuals and fully in their control.

  - Anyone with a dossier on you can blackmail you or rat you out.

- Something that is highly persuasive and centralized is a natural target for powerful entities to try to use as a leverage point.

- All of your context in one place allows it to unlock tons of value… which means it’s imperative that value works for you and not against you.

- We should have a private intelligence that tries to look out into the world for our goals

  - As opposed to an outside perspective from a company trying to see the world from our point of view.

- People are trying to figure out how to cram LLMs into the current generation of software.

  - What we need is the next generation of software for the AI era.

- Schemas over time in a system become more rigid.

  - The longer you've used it (the more data that is stored in it), and the more people that have used it, the more that the schema becomes impossible to change.

  - A schema that doesn't line up with what you want is a pain.

- Knowledge management systems today are a dead end.

  - Information goes in but doesn’t come out.

  - Knowledge management is an end in and of itself... but only for a small set of enthusiasts.

  - Notion workspaces always hit an asymptote.

    - Due to schemas not being flexible and requiring exponential cost for logarithmic benefit

  - A knowledge management tool that was open-ended and integrated with your life could be powerful.

  - Open-ended means if there's a feature that's missing, you can add the feature.

    - Perhaps with significant amounts of effort.

    - But in an open-ended system it is possible.

- Your software shouldn't optimize for some corporations' KPIs.

  - It should optimize for *your* KPIs.

  - This is the inversion of power that needs to happen in the era of AI.

- Users come for the primary use case.

  - They stay for the secondary use case.

- An interesting paper: [<u>Social Sycophancy: A Broader Understanding of LLM Sycophancy</u>](https://arxiv.org/abs/2505.13995)

  - [<u>Mike Caulfield points out</u>](https://archive.ph/URnTv) that giving these things personalities leads to the problem.

- Having a single unbiased LLM is an impossibility.

  - LLMs are an emergent cultural process, but with key influence points from the creators (intentionally or unintentionally).

  - Who should be the singular arbiters of the infinite, perfect ground truth?

    - There is no answer.

    - “Balancing” can itself also be bias.

      - A point someone made this week: “If one candidate's policy was to encourage infants to eat hot coals, and the other candidate's policy was to keep hot coals far from infants' mouths, would an LLM advocating for the latter be considered biased toward that candidate?"

  - So you need a diversity of models that people can switch between easily.

- Someone told me this week they sometimes scroll LinkedIn to get their infinite feed fix.

  - They don’t want to waste time on scrolling infinite feeds, so they configured their phone to not allow them to use Facebook or Twitter.

  - But the addiction to infinite feeds is so strong that he ended up scrolling LinkedIn.

  - That’s how you know you’ve got an addiction!

- There’s a difference between efficient and effective.

  - If you have a list of things to do but you don't know why it's there, then being efficient doesn't matter.

  - If an AI is scheduling tasks for you it feels like the opposite of effectiveness.

- In a world of increasing efficiency and cacophony, meaning is more important than ever.

- We work for the machines right now.

  - We should flip that.

  - The machines should help us live a more meaningful life.

- I want a lowercase-p productivity tool.

  - Not Productivity Porn for the most organized people.

  - A tool not trying to make me more productive, but trying to make the world more productive for me, so I can live a more purposeful, meaningful life.

  - Productivity for the rest of us.

  - Productivity is often "how can I be a better minion for the system that I'm embedded in".

  - Machines aren't serving us, we serve them.

  - How can we make it so the machines elevate us?

  - Orchestration, not productivity.

- Security and privacy are means to get control.

  - Privacy is a means to the power inversion.

  - The privacy is just in service of the power inversion.

- "ooohh!" to "...ewww" is what happens when you look closer at something that's only superficially resonant.

- The beginning of the web was an interplay of abstract vision and concrete implementation.

  - Ted Nelson had an abstract and idealized vision for the web but got frustrated trying to make it happen.

  - Tim Berners Lee was aware of the full vision, but created a very specific tool focused on the niche of documentation for a dozen physicists.

  - Pretending that it was smaller to build something viable.

  - But because it was open ended it could grow to eclipse the starting use case.

  - Tim understood the potential of the web, but didn't distract customers with it.

  - One of the big unlocks was having 404 pages.

    - Ted’s system wanted to never have broken links, which required bi-directional synchronization.

    - Tim added 404 pages, which acknowledged broken links would happen, and that was OK.

- The bigness of a vision can crowd out the concrete seed it will grow from.

  - When you have an open-ended system that will change the world, 70% of the perception of early adopters should be the simple starting part.

  - 20% of people should see the obvious extensions into direct adjacencies.

  - Only 10% get the whole abstract vision.

  - If all you talk about is the bigness of the vision, it snuffs out any concrete spark by not feeling big enough and being overwhelming.

- A dangerous phenomenon: sub-network metastasis.

  - This happens when a sub-component of a network grows much faster than other parts of the network.

  - It then grows to overshadow and encircle the rest of the network, choking it out and limiting its size.

  - The system now is captured by this sub-network, becoming ever more optimized for the sub-network, and ever worse for the rest of the network.

  - For example, Orkut was captured by the growth in Brazil, making it a less good fit everywhere else.

  - As another example, OnlyFans was captured by porn; now the “polite” use cases are crowded out.

  - The web’s early PMF was also about porn… but none of the front doors mentioned that and so it was possible for the polite parts of the network to also grow without being crowded out.

- When a means becomes an end, you are lost.

  - The means is urgent; the end is important.

  - Urgent things tend to overshadow important things.

- “Farming out” thinking is not thinking.

  - This is a mindset common in professors and judges.

  - They want to “farm out” thinking to some grad student or law clerk.

  - But critically, if anything comes back from the subordinate that disagrees with their existing mental model it will be rejected.

    - They’ll critique the underling as not having understood the idea properly.

  - It’s a one way process, not a two-way process, as true thinking is.

  - It’s impossible to get disconfirming evidence when you “farm out” thinking.

- Someone who is thinking 10 ply ahead will look to everyone else like they're making arbitrary decisions.

- Someone asked me why I publish Bits and Bobs each week.

  - The weekly reflection is the load bearing part.

  - The publishing is just the thing that makes me feel compelled to keep up the streak of spending time reflecting each week.

- This week I learned about [<u>ikigai</u>](https://www.japan.go.jp/kizuna/2022/03/ikigai_japanese_secret_to_a_joyful_life.html).

  - It’s the four interlocking dimensions of meaning to live a joyful life.

  - What you can be paid for.

  - What you are good at.

  - What the world needs.

  - What you love.

- There is no such thing as a fully open or fully closed system.

  - Open vs closed system is a matter of perspective.

  - A fish tank is a closed system... but part of an open system (the human who constructed it and puts food in).

  - An ocean is an open system, but effectively closed at the level of the planet.

- A concept I heard about this week that I like: "Type 2 recommender systems".

  - Recommends things that align with what you want to want, not what you want.

  - It echoes Harry Frankfurter's "Second order desires.”

- Everyone implicitly assumes that everyone else is thinking the same thing they are.

  - Because you're immersed in it, the water you swim in, you can't not see it that way.

  - So if you ever get distracted you forget that what other people see or know is totally different.

- I loved this essay from Aish about [<u>How to Know What to Do</u>](https://www.aishwaryadoingthings.com/how-to-know-what-to-do)

  - There’s a connection between alignment, intuition, and intention.

  - Applying presence to alignment gives intuition.

  - Applying noticing to intuition gives intention.

  - Applying execution to intention gives alignment.

- A dictator could in theory be great if they were competent and aligned with your interests.

  - But they’re terrible if incompetent or not aligned with your interests.

  - Perfect competence or perfect alignment is impossible.

    - To say nothing of everyone’s interests being different.

  - So dictators are fundamentally terrible.

- Any metric must be a proxy.

  - That’s because it has to be a model of reality, not reality, to be operationalized.

  - Otherwise there's no compression, no leverage.

  - A map that is 1:1 with the territory is not useful.

  - But the fact that a map is not 1:1 with the territory means there’s a difference between what you’re measuring and what you care about.

  - Swarms that are optimizing for the metric will exploit that difference.

- Prosocial things bring you more into the world.

  - Antisocial things take you out of the world.

- I believe in the power of emergence.

  - That is, that emergence exists, and that it's an important, powerful force.

  - Emergence means "the whole is greater than the sum of the parts".

    - Because the way the parts *relate* matters, not just the parts itself.

    - [<u>Ursus Wehrli’s images</u>](https://belopotosky.wordpress.com/wp-content/uploads/2011/09/ursus04.gif) that tidy the components of complex things show that emergence is real, and objects are more than just the sum of their parts.

  - That, more than being a "systems" guy, is what differentiates Radagasts from Sarumans.

  - To be clear, emergence is not just a thing I believe in, it is also obviously, incontrovertibly true.

  - It’s just something that Sarumans don’t pay much attention to, or is invisible to them.